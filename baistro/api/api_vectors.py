from typing import Union

import numpy as np
from PIL import Image
from apiflask import APIFlask
from sentence_transformers import util
from torch import Tensor
from baistro.api.schemas import VectorResponse, VectorQueryResponse, VectorRequest, VectorFileRequest, VectorQueryRequest, VectorBatchResponse, VectorBatchRequest
from baistro._boot import Services
from baistro.model_control.infer_result import InferTracker
from baistro.model_control.model_base import model_url, ModelBase
from baistro.model_control.models import models
from baistro.models.vector_code import VectorCodeModel
from baistro.models.vector_image import VectorImageModel
from baistro.models.vector_text import VectorTextModel


def get_input(r):
    if r is None:
        raise ValueError('No request data.')
    if 'file' in r:
        # todo: support batch files
        return Image.open(r['file'])
    # todo: support data and files
    return r['input']


def api_vectors(app: APIFlask, s: Services):
    def handle_vector_query(model_id: str, json_data):
        infer_res = InferTracker()
        m, tracker = models.get_tracked(model_id, infer_res)
        options = json_data.get('options', {})

        query = json_data['query']
        context = json_data['context']
        if not isinstance(context, list):
            return {'error': 'context must be a list of strings'}, 400

        on_encoded_context = tracker('encode_context')
        used_tokens0, context_emb = m.encode(context)
        on_encoded_context(tokens=used_tokens0)

        on_encoded_query = tracker('encode_query')
        used_tokens1, query_emb = m.encode(query)
        on_encoded_query(tokens=used_tokens1)

        max_top = min(options.get('top', 1), len(context))
        hits = util.semantic_search(query_emb, context_emb, top_k=max_top)[0]
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
            '_usages': infer_res.usages,
            'matches': top_matches,
        }

    def handle_vector(model_id: str, json_data):
        infer_res = InferTracker()
        m, tracker = models.get_tracked(model_id, infer_res)

        on_processed = tracker('infer')
        used_tokens, vc = m.encode(get_input(json_data))
        on_processed(tokens=used_tokens)
        return {
            '_usages': infer_res.usages,
            'embeddings': vc.numpy().tolist() if isinstance(vc, Tensor) else vc.astype(np.float32).tolist(),
        }

    # todo: merge batch endpoints or find another solution, there is no native Union field in APIFlask
    #       note: even if found a solution for `.input`, the `.output` is still dumbed by schema
    @app.route(f'/{VectorTextModel.id}', methods=['POST'])
    @app.input(VectorRequest, schema_name=f'VectorRequest')
    @app.output(VectorResponse())
    @app.doc(
        tags=['vector-text'],
        operation_id='vector_text',
        description=f'''
Active model: {active_model_md(VectorTextModel)}

Model env var: `MODEL__VECTOR_TEXT`

Recommended models:

- [all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1)
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
        ''',
    )
    def vector_text(json_data):
        return handle_vector(VectorTextModel.id, json_data)

    @app.route(f'/{VectorTextModel.id}-batch', methods=['POST'])
    @app.input(VectorBatchRequest, schema_name=f'VectorBatchRequest')
    @app.output(VectorBatchResponse())
    @app.doc(
        tags=['vector-text'],
        description=f'''
See `vector_text` for more, this is the batch encoding endpoint for it.
        ''',
    )
    def vector_text_batch(json_data):
        return handle_vector(VectorTextModel.id, json_data)

    @app.route(f'/{VectorTextModel.id}/query', methods=['POST'])
    @app.input(VectorQueryRequest, schema_name=f'VectorQueryRequest')
    @app.output(VectorQueryResponse())
    @app.doc(
        tags=['vector-text'],
        description=f'''
See `vector_text` for more, this is the semantic query endpoint for it.
        ''',
    )
    def vector_text__query(json_data):
        return handle_vector_query(VectorTextModel.id, json_data)

    @app.route(f'/{VectorCodeModel.id}', methods=['POST'])
    @app.input(VectorRequest, schema_name=f'VectorRequest')
    @app.output(VectorResponse())
    @app.doc(
        tags=['vector-code'],
        operation_id='vector_code',
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
    @app.input(VectorBatchRequest, schema_name=f'VectorBatchRequest')
    @app.output(VectorBatchResponse())
    @app.doc(
        tags=['vector-code'],
        description=f'''
See `vector_code` for more, this is the batch encoding endpoint for it.
        ''',
    )
    def vector_code_batch(json_data):
        return handle_vector(VectorCodeModel.id, json_data)

    @app.route(f'/{VectorCodeModel.id}/query', methods=['POST'])
    @app.input(VectorQueryRequest, schema_name=f'VectorQueryRequest')
    @app.output(VectorQueryResponse())
    @app.doc(
        tags=['vector-code'],
        description=f'''
See `vector_code` for more, this is the semantic query endpoint for it.
        ''',
    )
    def vector_code__query(json_data):
        return handle_vector_query(VectorCodeModel.id, json_data)

    @app.route(f'/{VectorImageModel.id}-input', methods=['POST'])
    @app.input(VectorRequest, schema_name=f'VectorRequest', location='json_or_form')
    @app.output(VectorResponse())
    @app.doc(
        tags=['vector-image'],
        description=f'''
See `vector_image` for more information.
        ''',
    )
    def vector_image_input(json_or_form_data):
        infer_res = InferTracker()
        m, tracker = models.get_tracked(VectorImageModel.id, infer_res)
        on_processed = tracker('infer')
        used_tokens, vc = m.encode(get_input(json_or_form_data))
        on_processed(tokens=used_tokens)
        return {
            '_usages': infer_res.usages,
            'embeddings': vc.numpy().tolist() if isinstance(vc, Tensor) else vc.astype(np.float32).tolist(),
        }

    # todo: APIFlask does not support "json or form with files", either no Input or no File support
    #       routes should be merged once supported https://github.com/apiflask/apiflask/issues/603
    @app.route(f'/{VectorImageModel.id}', methods=['POST'])
    # @app.input(VectorFileRequest, schema_name=f'VectorFileRequest', location='json_or_form_and_files')
    @app.input(VectorFileRequest, schema_name=f'VectorFileRequest', location='form_and_files')
    @app.output(VectorResponse())
    @app.doc(
        tags=['vector-image'],
        operation_id='vector_image',
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
        used_tokens, vc = m.encode(get_input(form_and_files_data))
        on_processed(tokens=used_tokens)
        return {
            '_usages': infer_res.usages,
            'embeddings': vc.numpy().tolist() if isinstance(vc, Tensor) else vc.astype(np.float32).tolist(),
        }


def active_model_md(model: Union[ModelBase, type[ModelBase]]):
    url = model_url(VectorImageModel)
    if url:
        return f'[{model.name}]({url})'
    # return f'{model.name} from `{model.folder}`'
    return f'{model.name}'
