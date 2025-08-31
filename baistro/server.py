import signal
import sys
import logging
from pathlib import Path

from apiflask import APIFlask, Schema, fields
from apispec.ext.marshmallow.field_converter import _VALID_PROPERTIES

from baistro._boot import boot
from baistro.api.api import ai_api
from baistro.config.config import AppConfig

from baistro.helper import ts
from flask import render_template, url_for
from flask_cors import CORS

from baistro.model_control.models import models
from baistro.models.vector_code import VectorCodeModel
from baistro.models.vector_image import VectorImageModel
from baistro.models.vector_text import VectorTextModel

# todo: https://stackoverflow.com/a/16993115/2073149
logging.basicConfig(stream=sys.stdout, level=AppConfig.LOGGING_LEVEL)
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

_VALID_PROPERTIES.add('widget')

s = boot()

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
    version='0.2.2',
    docs_ui=AppConfig.API_DOCS_UI,
)
app.config['OPENAPI_VERSION'] = '3.1.0'

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
