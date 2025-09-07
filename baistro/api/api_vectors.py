from typing import Union, Dict, List, Tuple

import numpy as np
from PIL import Image
from apiflask import APIFlask, fields, Schema, blueprint
from marshmallow import validates_schema, ValidationError
from marshmallow.validate import Length, OneOf
from sentence_transformers import util
from sentence_transformers.util import cos_sim, dot_score, euclidean_sim, manhattan_sim
from torch import Tensor

from baistro.api.api_models import get_folder_size
from baistro.api.schemas import InferBaseResponse
from baistro._boot import Services
from baistro.model_control.infer_result import InferTracker
from baistro.model_control.model_base import model_url, ModelBase
from baistro.model_control.models import models
from baistro.models.vector_code import VectorCodeModel
from baistro.models.vector_image import VectorImageModel
from baistro.models.vector_text import VectorTextModel

_score_functions = {
    'cosine': cos_sim,
    'dot_score': dot_score,
    'euclidean': euclidean_sim,
    'manhattan': manhattan_sim,
}


class VectorRequest(Schema):
    input = fields.String(required=True)


class VectorBatchRequest(Schema):
    input = fields.List(fields.String(), required=True, validate=Length(1))


class VectorQueryOptions(Schema):
    top = fields.Integer(metadata={'example': 3})
    min_score = fields.Float(metadata={'default': 0.1})
    scorer = fields.String(
        metadata={'default': 'cosine'},
        validate=OneOf(list(_score_functions.keys())),
    )


class VectorSearchRequest(Schema):
    options = fields.Nested(VectorQueryOptions())
    # todo: implement batch query, if not just broken with schema
    query = fields.String(required=True)
    context = fields.List(fields.String(), required=True)


class VectorFileRequest(Schema):
    file = fields.File()
    input = fields.String()

    # todo: either file or input is required, but found no way to add it on Schema
    # def __init__(self, **kwargs):
    #     metadata = {
    #         "anyOf": [
    #             {"required": ["file"]},
    #             {"required": ["input"]}
    #         ]
    #     }
    #
    #     if 'metadata' in kwargs:
    #         kwargs['metadata'] = {**metadata, **kwargs['metadata']}
    #     else:
    #         kwargs['metadata'] = metadata
    #
    #     super().__init__(**kwargs)

    @validates_schema
    def validate_file_or_input(self, data, **kwargs):
        if not data.get("file") and not data.get("input"):
            raise ValidationError("Either 'file' or 'input' is required.", field_name="file")
        if data.get("file") and data.get("input"):
            raise ValidationError("Only one of 'file' or 'input' should be provided.")


class VectorResponse(InferBaseResponse):
    embeddings = fields.List(fields.Number())


class VectorBatchResponse(InferBaseResponse):
    embeddings = fields.List(fields.List(fields.Number()))


class VectorQueryMatch(Schema):
    match = fields.String()
    score = fields.Float()


class VectorSearchResponse(InferBaseResponse):
    matches = fields.List(fields.Nested(VectorQueryMatch()))


class VectorParaphraseOptions(Schema):
    min_score = fields.Float(metadata={'default': 0.6})


class VectorParaphraseRequest(Schema):
    input = fields.List(
        fields.String(), required=True, validate=Length(1),
        metadata={
            'examples': [
                [
                    "I enjoy watching action movies on the weekend.",
                    "On weekends, I like to watch action films.",
                    "Thriller movies always keep me on the edge of my seat.",
                    "Watching thrillers makes me feel tense and excited.",
                    "I rarely eat fast food because I prefer home-cooked meals.",
                    "Homemade meals are better than fast food, so I avoid the latter.",
                    "I love trying new types of cuisine whenever I travel.",
                    "Exploring different cuisines is one of my favorite travel activities.",
                    "Watching a movie while eating popcorn is my favorite pastime.",
                    "My favorite way to watch films is with a big bowl of popcorn.",
                    "Romantic comedies usually feel predictable to me.",
                    "I often find romantic comedy films too predictable and formulaic."
                ],
                [
                    "The cat sits outside",
                    "A man is playing guitar",
                    "I love pasta",
                    "The new movie is awesome",
                    "The cat plays in the garden",
                    "A woman watches TV",
                    "The new movie is so great",
                    "Do you like pizza?"
                ]
            ]
        }
    )
    options = fields.Nested(VectorParaphraseOptions())


class VectorParaphraseResponse(InferBaseResponse):
    paraphrases = fields.List(fields.List(fields.Raw()))


def get_input(r):
    if r is None:
        raise ValueError('No request data.')
    if 'file' in r:
        # todo: support batch files
        return Image.open(r['file'])
    # todo: support data and files
    return r['input']


def handle_vector_search(model_id: str, json_data):
    infer_res = InferTracker()
    m, tracker = models.get_tracked(model_id, infer_res)
    options = json_data.get('options', {})

    query = json_data['query']
    context = json_data['context']
    if not isinstance(context, list):
        return {'error': 'context must be a list of strings'}, 400

    on_encoded_context = tracker('encode_context')
    used_tokens0, context_emb = m.encode_with_stats(context)
    on_encoded_context(tokens=used_tokens0)

    on_encoded_query = tracker('encode_query')
    used_tokens1, query_emb = m.encode_with_stats(query)
    on_encoded_query(tokens=used_tokens1)

    max_top = min(options.get('top', 1), len(context))
    score_function_name = options.get('scorer', 'cosine')

    score_function = _score_functions.get(score_function_name, cos_sim)
    hits = util.semantic_search(
        query_emb,
        context_emb,
        top_k=max_top,
        score_function=score_function,
    )[0]
    min_score = options.get('min_score', 0.1)
    top_matches = []
    for i in range(max_top):
        hit = hits[i]
        if min_score > hit['score']:
            break
        top_matches.append({
            'match': context[hit['corpus_id']],
            'score': float(hit['score']),
        })

    return {
        'usage': infer_res.usage,
        'matches': top_matches,
    }


def handle_vector_paraphrase(model_id: str, json_data):
    infer_res = InferTracker()
    m, tracker = models.get_tracked(model_id, infer_res)
    options = json_data.get('options', {})
    min_score = options.get('min_score', 0.6)

    sentences = json_data['input']
    if not isinstance(sentences, list):
        return {'error': 'input must be a list of strings'}, 400

    on_mine_paraphrases = tracker('mine_paraphrases')
    paraphrases_scores = util.paraphrase_mining(m, sentences)
    on_mine_paraphrases()

    # todo: using numeric id-references would be better/smaller, but not so nice when trying it manually
    paraphrases: List[Tuple[str, str, float]] = []

    for paraphrase in paraphrases_scores:
        score, i, j = paraphrase
        if score >= min_score:
            paraphrases.append((sentences[i], sentences[j], float(score)))

    return {
        'usage': infer_res.usage,
        'paraphrases': paraphrases,
    }


def handle_vector(model_id: str, json_data):
    infer_res = InferTracker()
    m, tracker = models.get_tracked(model_id, infer_res)

    on_processed = tracker('infer')
    used_tokens, vc = m.encode_with_stats(get_input(json_data))
    on_processed(tokens=used_tokens)
    return {
        'usage': infer_res.usage,
        'embeddings': vc.numpy().tolist() if isinstance(vc, Tensor) else vc.astype(np.float32).tolist(),
    }


def api_vectors(app: APIFlask, s: Services):
    # filter the result of `models.list()` for models with `vector` in `tasks`
    # ignoring legacy vector model files
    vector_models = [m for m in models.list() if 'vector' in m.tasks and not (m == VectorCodeModel or m == VectorImageModel or m == VectorTextModel)]

    for vector_model in vector_models:
        vector_model_id = vector_model.id
        vector_model_tag = vector_model_id if vector_model_id.startswith('vector-') else f'vector-{vector_model_id}'

        # factory function to prevent python late binding issues
        def make_vector_endpoints(vector_model_id=vector_model_id, vector_model=vector_model):
            # todo: depending on modality, the routes and schema are different! e.g. image model needs file upload and has no batch support atm.
            @app.route(f'/vector/{vector_model_id}', methods=['GET'], endpoint=f'vector-{vector_model_id}-info')
            @app.doc(
                tags=['Models', vector_model_tag],
                operation_id=f'vector_{vector_model_id}',
                description=f'''
This endpoint provides information about the `{vector_model_id}` vector model. It also indicates the local folder where the model is stored.

This information can be used to understand the capabilities and characteristics of the model before making encoding or search requests.

Active model: {active_model_md(vector_model)}
''',
            )
            def vector_model_info():
                return {
                    'id': vector_model_id,
                    'name': vector_model.name,
                    'description': vector_model.description,
                    'url': model_url(vector_model),
                    'modality': vector_model.modality,
                    'tasks': vector_model.tasks,
                    'features': vector_model.features,
                    # 'license': vector_model.license,
                    # 'citation': vector_model.citation,
                    'folder': vector_model.folder,
                    'size': get_folder_size(vector_model.folder),
                }

            @app.route(f'/vector/{vector_model_id}/encode', methods=['POST'], endpoint=f'vector-{vector_model_id}-encode')
            @app.input(VectorRequest)
            @app.output(VectorResponse())
            @app.doc(
                tags=[vector_model_tag],
                description=f'''
See `{vector_model_id}` for more, this is the encoding endpoint for it.
''',
            )
            def vector_model_encode(json_data):
                return handle_vector(vector_model_id, json_data)

            if 'image' not in vector_model.modality:
                @app.route(f'/vector/{vector_model_id}/encode-batch', methods=['POST'], endpoint=f'vector-{vector_model_id}-batch')
                @app.input(VectorBatchRequest)
                @app.output(VectorBatchResponse())
                @app.doc(
                    tags=[vector_model_tag],
                    description=f'''
See `{vector_model_id}` for more, this is the batch encoding endpoint for it.
''',
                )
                def vector_model_batch_encode(json_data):
                    return handle_vector(vector_model_id, json_data)

                if vector_model.features and 'search' in vector_model.features:
                    @app.route(f'/vector/{vector_model_id}/search', methods=['POST'], endpoint=f'vector-{vector_model_id}-search')
                    @app.input(VectorSearchRequest)
                    @app.output(VectorSearchResponse())
                    @app.doc(
                        tags=[vector_model_tag],
                        description=f'''
See `{vector_model_id}` for more, this is the semantic query endpoint for it.
''',
                    )
                    def vector_model_search(json_data):
                        return handle_vector_search(vector_model_id, json_data)

                if vector_model.features and 'paraphrase' in vector_model.features:
                    @app.route(f'/vector/{vector_model_id}/paraphrase', methods=['POST'], endpoint=f'vector-{vector_model_id}-paraphrase')
                    @app.input(VectorParaphraseRequest)
                    @app.output(VectorParaphraseResponse())
                    @app.doc(
                        tags=[vector_model_tag],
                        description=f'''
See `{vector_model_id}` for more, this is the paraphrase mining endpoint for it.
''',
                    )
                    def vector_model_paraphrase(json_data):
                        return handle_vector_paraphrase(vector_model_id, json_data)

            if 'image' in vector_model.modality:
                @app.route(f'/vector/{vector_model_id}/encode-file', methods=['POST'], endpoint=f'vector-{vector_model_id}-encode-file')
                @app.input(VectorFileRequest, location='form_and_files')
                @app.output(VectorResponse())
                @app.doc(
                    tags=[vector_model_tag],
                    description=f'''
See `{vector_model_id}` for more, this is the file upload endpoint for it.
''',
                )
                def vector_model_encode_image(form_and_files_data):
                    return handle_vector(vector_model_id, form_and_files_data)

        make_vector_endpoints()

    # todo: merge batch endpoints or find another solution, there is no native Union field in APIFlask
    #       note: even if found a solution for `.input`, the `.output` is still dumbed by schema

    @app.route(f'/{VectorTextModel.id}', methods=['POST'])
    @app.input(VectorRequest)
    @app.output(VectorResponse())
    @app.doc(
        tags=['vector-text'],
        operation_id='vector_text',
        deprecated=True,
        description=f'''
Active model: {active_model_md(VectorTextModel)}

Model env var: `MODEL__VECTOR_TEXT`

Recommended models:

- [all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1)
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [distiluse-base-multilingual-cased-v1](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1)
- [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)
''',
    )
    def vector_text(json_data):
        return handle_vector(VectorTextModel.id, json_data)

    @app.route(f'/{VectorTextModel.id}-batch', methods=['POST'])
    @app.input(VectorBatchRequest)
    @app.output(VectorBatchResponse())
    @app.doc(
        tags=['vector-text'],
        deprecated=True,
        description=f'''
See `vector_text` for more, this is the batch encoding endpoint for it.
''',
    )
    def vector_text_batch(json_data):
        return handle_vector(VectorTextModel.id, json_data)

    @app.route(f'/{VectorTextModel.id}/query', methods=['POST'])
    @app.input(VectorSearchRequest)
    @app.output(VectorSearchResponse())
    @app.doc(
        tags=['vector-text'],
        deprecated=True,
        description=f'''
See `vector_text` for more, this is the semantic query endpoint for it.
''',
    )
    def vector_text__query(json_data):
        return handle_vector_search(VectorTextModel.id, json_data)

    @app.route(f'/{VectorCodeModel.id}', methods=['POST'])
    @app.input(VectorRequest)
    @app.output(VectorResponse())
    @app.doc(
        tags=['vector-code'],
        operation_id='vector_code',
        deprecated=True,
        description=f'''
Active model: {active_model_md(VectorCodeModel)}

Model env var: `MODEL__VECTOR_CODE`

Recommended models:

- [st-codesearch-distilroberta-base](https://huggingface.co/flax-sentence-embeddings/st-codesearch-distilroberta-base)
- [unixcoder-base](https://huggingface.co/microsoft/unixcoder-base)
''',
    )
    def vector_code(json_data):
        return handle_vector(VectorCodeModel.id, json_data)

    @app.route(f'/{VectorCodeModel.id}-batch', methods=['POST'])
    @app.input(VectorBatchRequest)
    @app.output(VectorBatchResponse())
    @app.doc(
        tags=['vector-code'],
        deprecated=True,
        description=f'''
See `vector_code` for more, this is the batch encoding endpoint for it.
''',
    )
    def vector_code_batch(json_data):
        return handle_vector(VectorCodeModel.id, json_data)

    @app.route(f'/{VectorCodeModel.id}/query', methods=['POST'])
    @app.input(VectorSearchRequest)
    @app.output(VectorSearchResponse())
    @app.doc(
        tags=['vector-code'],
        deprecated=True,
        description=f'''
See `vector_code` for more, this is the semantic query endpoint for it.
''',
    )
    def vector_code__query(json_data):
        return handle_vector_search(VectorCodeModel.id, json_data)

    @app.route(f'/{VectorImageModel.id}-input', methods=['POST'])
    @app.input(VectorRequest, location='json_or_form')
    @app.output(VectorResponse())
    @app.doc(
        tags=['vector-image'],
        deprecated=True,
        description=f'''
See `vector_image` for more information.
''',
    )
    def vector_image_input(json_or_form_data):
        infer_res = InferTracker()
        m, tracker = models.get_tracked(VectorImageModel.id, infer_res)
        on_processed = tracker('infer')
        used_tokens, vc = m.encode_with_stats(get_input(json_or_form_data))
        on_processed(tokens=used_tokens)
        return {
            'usage': infer_res.usage,
            'embeddings': vc.numpy().tolist() if isinstance(vc, Tensor) else vc.astype(np.float32).tolist(),
        }

    # todo: APIFlask does not support "json or form with files", either no Input or no File support
    #       routes should be merged once supported https://github.com/apiflask/apiflask/issues/603
    @app.route(f'/{VectorImageModel.id}', methods=['POST'])
    # @app.input(VectorFileRequest, schema_name=f'VectorFileRequest', location='json_or_form_and_files')
    @app.input(VectorFileRequest, location='form_and_files')
    @app.output(VectorResponse())
    @app.doc(
        tags=['vector-image'],
        operation_id='vector_image',
        deprecated=True,
        description=f'''
Active model: {active_model_md(VectorImageModel)}

Model env var: `MODEL__VECTOR_IMAGE`

Recommended models:

- [clip-ViT-B-32](https://huggingface.co/sentence-transformers/clip-ViT-B-32)
- [clip-ViT-B-32-multilingual-v1](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1)
'''
    )
    def vector_image(form_and_files_data):
        infer_res = InferTracker()
        m, tracker = models.get_tracked(VectorImageModel.id, infer_res)
        on_processed = tracker('infer')
        used_tokens, vc = m.encode_with_stats(get_input(form_and_files_data))
        on_processed(tokens=used_tokens)
        return {
            'usage': infer_res.usage,
            'embeddings': vc.numpy().tolist() if isinstance(vc, Tensor) else vc.astype(np.float32).tolist(),
        }


def active_model_md(model: Union[ModelBase, type[ModelBase]]):
    url = model_url(VectorImageModel)
    if url:
        return f'[{model.name}]({url})'
    # return f'{model.name} from `{model.folder}`'
    return f'{model.name}'
