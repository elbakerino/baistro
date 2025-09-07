import signal
import sys
import logging
from pathlib import Path

from apiflask import APIFlask, Schema, fields
from apispec.ext.marshmallow.field_converter import _VALID_PROPERTIES

from baistro._boot import boot
from baistro.config.config import AppConfig

from baistro.helper import ts
from flask import render_template, url_for
from flask_cors import CORS

from baistro.model_control.models import models
from baistro.models.vector_code import VectorCodeModel
from baistro.models.vector_image import VectorImageModel
from baistro.models.vector_text import VectorTextModel

from baistro.api.api import ai_api

# todo: https://stackoverflow.com/a/16993115/2073149
logging.basicConfig(stream=sys.stdout, level=AppConfig.LOGGING_LEVEL)
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.INFO)

_VALID_PROPERTIES.add('widget')
_VALID_PROPERTIES.add('examples')
_VALID_PROPERTIES.add('uniqueItems')

s = boot()

for model_id in models.models_preload:
    model = models.get(model_id)
    if model.model is None:
        raise RuntimeError(f"Model {model_id} did not initialize")

    logging.info(f'preloaded {model_id} with {model.name}')

if AppConfig.PRELOAD_VECTOR_TEXT:
    model = models.get(VectorTextModel.id)
    if model.model is None:
        raise RuntimeError("VectorTextModel.model did not initialize")

    logging.info(f'preloaded {VectorTextModel.id} with {model.name}')

if AppConfig.PRELOAD_VECTOR_CODE:
    model = models.get(VectorCodeModel.id)
    if model.model is None:
        raise RuntimeError("VectorCodeModel.model did not initialize")

    logging.info(f'preloaded {VectorCodeModel.id} with {model.name}')

if AppConfig.PRELOAD_VECTOR_IMAGE:
    model = models.get(VectorImageModel.id)
    if model.model is None:
        raise RuntimeError("VectorImageModel.model did not initialize")

    logging.info(f'preloaded {VectorImageModel.id} with {model.name}')

app = APIFlask(
    __name__,
    title='baistro',
    version='0.4.0',
    docs_ui=AppConfig.API_DOCS_UI,
)
app.config['OPENAPI_VERSION'] = '3.1.0'
app.config['TAGS'] = [
    {'name': 'NLP', 'description': 'APIs for Natural Language Processing, including locale identification, sentence segmentation, and various classification tasks like sentiment analysis and named entity recognition.'},
    {'name': 'Models', 'description': 'Information about defined models and their availability.'},
    {'name': 'vector-text', 'description': 'Vector embeddings for plaintext, the foundation for many AI tasks like semantic search, clustering, and classification. These embeddings capture the semantic meaning of text, enabling advanced semantic understanding, similarity comparisons, and advanced retrieval augmented generation (RAG) systems.'},
    {'name': 'vector-code', 'description': 'Vector embeddings for source code, use it for semantic search on code snippets (search code by natural language) and finding similar code.'},
    {'name': 'vector-image', 'description': 'Vector embeddings for images, use it for search-by-image and similar-image-search. These embeddings are generated using advanced deep learning models like CLIP, which learn to associate images with their textual descriptions. This enables cross-modal search capabilities, allowing users to find images using text queries and vice-versa.'},
]
app.config['INFO'] = {
    'description': 'Some APIs for AI models that run on CPU, designed for deterministic NLP. Includes Stanza and Sentence Transformers to provide a self-contained NLP processing unit for common tasks.',
    'license': {
        'name': 'MIT License',
        'url': 'https://github.com/elbakerino/baistro/blob/main/LICENSE',
    }
}
app.config['EXTERNAL_DOCS'] = {
    'description': 'baistro repository',
    'url': 'https://github.com/elbakerino/baistro',
}

# todo: support configuring server and related openapi/flask config like base-path
# app.config['SERVERS'] = ''

# app.config['SYNC_LOCAL_SPEC'] = True
# app.config['LOCAL_SPEC_PATH'] = 'openapi.json'

CORS(
    app,
    origins=AppConfig.CORS_ORIGINS,
    send_wildcard=AppConfig.CORS_SEND_WILDCARD,
)

original_sigint_handler = signal.getsignal(signal.SIGINT)


def on_signal(signal_number, frame):
    # needed for flask, calls the org. signal handler but still does not gracefully wait
    if original_sigint_handler:
        original_sigint_handler(signal_number, frame)


# signal.signal(signal.SIGKILL, on_signal)
signal.signal(signal.SIGTERM, on_signal)
signal.signal(signal.SIGINT, on_signal)

CSS_CONTENT = (Path(app.root_path) / "templates" / "main-dark.css").read_text(encoding="utf-8")


@app.route('/')
def route_home():
    links = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint == 'static':
            continue
        url = url_for(rule.endpoint, **(rule.defaults or {}))
        methods = rule.methods.copy()
        if 'HEAD' in methods:
            methods.remove('HEAD')
        if 'OPTIONS' in methods:
            methods.remove('OPTIONS')
        links.append((url, list(methods), rule.endpoint, rule.arguments, rule.defaults))

    return render_template(
        'index.html',
        version=AppConfig.APP_ENV,
        links=links,
        css=CSS_CONTENT
    )


class PingResponse(Schema):
    now = fields.String()


@app.route('/ping')
@app.output(PingResponse)
def route_ping():
    return {
        "now": ts.now_iso(micros=False),
    }


ai_api(app, s)
