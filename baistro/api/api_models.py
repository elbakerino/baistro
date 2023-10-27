import os

import numpy as np
from PIL import Image
from flask import Flask, request
from sentence_transformers import util
from torch import Tensor

from baistro._boot import Services
from baistro.helper.qag_parser import qag_split_pairs
from baistro.model_control.infer_result import InferTracker
from baistro.model_control.model_base import model_url
from baistro.model_control.models import models
from baistro.models.dit_base import DitBaseModel
from baistro.models.dit_large import DitLargeModel
from baistro.models.donut_docvqa import DonutDocvqaModel
from baistro.models.donut_to_data import DonutToDataModel
from baistro.models.donut_to_text import DonutToTextModel
from baistro.models.qag_en_base import QagEnBaseModel
from baistro.models.qnli_en_base import QnliEnBaseModel
from baistro.models.vector_code import VectorCodeModel
from baistro.models.vector_image import VectorImageModel
from baistro.models.vector_text import VectorTextModel
from baistro.models.qa_en_large import QaEnLargeModel
from baistro.models.qa_en_base import QaEnBaseModel


def get_folder_size(folder_path):
    total_size = 0
    for path, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(path, file)
            total_size += os.path.getsize(file_path)
    return total_size


def get_input(r):
    if 'file' in r.files:
        # todo: support batch files
        return Image.open(r.files['file'])
    # todo: support data and files
    return r.json['input']


def api_models(app: Flask, s: Services):
    @app.route('/models', methods=['GET'])
    def model_list():
        models_stats = {}

        for model in models.list():
            models_stats[model.id] = {
                'tasks': model.tasks,
                'name': model.name,
                'locale': model.locale if hasattr(model, 'locale') else None,
                'url': model_url(model),
                'size': get_folder_size(model.folder),
            }
        return {
            'total': len(models_stats),
            'stats': models_stats,
        }

    @app.route(f'/{QagEnBaseModel.id}', methods=['POST'])
    def qag_en_base():
        infer_res = InferTracker()
        m, tracker = models.get_tracked(QagEnBaseModel.id, infer_res)

        input = request.json['input']

        on_processed = tracker('infer')
        used_tokens, result = m.generate('generate question and answer: ' + input)
        on_processed(tokens=used_tokens)
        pairs = qag_split_pairs(result)
        return {
            '_usages': infer_res.usages,
            'outcome': pairs,
        }

    @app.route(f'/{QnliEnBaseModel.id}', methods=['POST'])
    def qnli_en_base():
        infer_res = InferTracker()
        m, tracker = models.get_tracked(QnliEnBaseModel.id, infer_res)

        input = request.json['input']

        on_processed = tracker('infer')
        used_tokens, result = m.generate(input)
        on_processed(tokens=used_tokens)
        return {
            '_usages': infer_res.usages,
            'outcome': result,
        }

    @app.route(f'/{QaEnBaseModel.id}', methods=['POST'])
    def qa_en_base():
        infer_res = InferTracker()
        m, tracker = models.get_tracked(QaEnBaseModel.id, infer_res)

        query = request.json['query']
        context = request.json['context']

        on_processed = tracker('infer')
        used_tokens, result = m.generate(query, context)
        on_processed(tokens=used_tokens)
        return {
            '_usages': infer_res.usages,
            'outcome': result,
        }

    @app.route(f'/{QaEnLargeModel.id}', methods=['POST'])
    def qa_en_large():
        infer_res = InferTracker()
        m, tracker = models.get_tracked(QaEnLargeModel.id, infer_res)

        query = request.json['query']
        context = request.json['context']

        on_processed = tracker('infer')
        used_tokens, result = m.generate(query, context)
        on_processed(tokens=used_tokens)
        return {
            '_usages': infer_res.usages,
            'outcome': result,
        }

    def handle_vqa(model_id):
        infer_res = InferTracker()
        m, tracker = models.get_tracked(model_id, infer_res)

        file = request.files.get('file')
        if not file:
            return {'error': 'Missing "file"'}, 400
        query = request.form.get('query')
        if not query:
            return {'error': 'Missing "query"'}, 400

        img = Image.open(file)
        on_processed = tracker('infer')
        used_tokens, result = m.generate(query, img)
        on_processed(tokens=used_tokens)
        return {
            '_usages': infer_res.usages,
            'outcome': result,
        }

    @app.route(f'/{DonutDocvqaModel.id}', methods=['POST'])
    def donut_docvqa():
        return handle_vqa(DonutDocvqaModel.id)

    @app.route(f'/{DonutToDataModel.id}', methods=['POST'])
    def donut_to_data():
        infer_res = InferTracker()
        m, tracker = models.get_tracked(DonutToDataModel.id, infer_res)

        file = request.files.get('file')
        if not file:
            return {'error': 'Missing "file"'}, 400

        img = Image.open(file)
        on_processed = tracker('infer')
        used_tokens, result = m.generate(img)
        on_processed(tokens=used_tokens)
        return {
            '_usages': infer_res.usages,
            'outcome': result,
        }

    @app.route(f'/{DonutToTextModel.id}', methods=['POST'])
    def donut_to_text():
        infer_res = InferTracker()
        m, tracker = models.get_tracked(DonutToTextModel.id, infer_res)

        file = request.files.get('file')
        if not file:
            return {'error': 'Missing "file"'}, 400

        img = Image.open(file)
        on_processed = tracker('infer')
        used_tokens, result = m.generate(img)
        on_processed(tokens=used_tokens)
        return {
            '_usages': infer_res.usages,
            'outcome': result,
        }

    @app.route(f'/{DitBaseModel.id}', methods=['POST'])
    def dit_base():
        infer_res = InferTracker()
        m, tracker = models.get_tracked(DitBaseModel.id, infer_res)

        file = request.files.get('file')
        if not file:
            return {'error': 'Missing "file"'}, 400

        img = Image.open(file)
        on_processed = tracker('infer')
        used_tokens, result = m.generate(img)
        on_processed(tokens=used_tokens)
        return {
            '_usages': infer_res.usages,
            'outcome': result,
        }

    @app.route(f'/{DitLargeModel.id}', methods=['POST'])
    def dit_large():
        infer_res = InferTracker()
        m, tracker = models.get_tracked(DitLargeModel.id, infer_res)

        file = request.files.get('file')
        if not file:
            return {'error': 'Missing "file"'}, 400

        img = Image.open(file)
        on_processed = tracker('infer')
        used_tokens, result = m.generate(img)
        on_processed(tokens=used_tokens)
        return {
            '_usages': infer_res.usages,
            'outcome': result,
        }

    def handle_vector_query(model_id: str):
        infer_res = InferTracker()
        m, tracker = models.get_tracked(model_id, infer_res)
        options = request.json.get('options', {})

        query = request.json['query']
        context = request.json['context']
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

    @app.route(f'/{VectorTextModel.id}', methods=['POST'])
    def vector_text():
        infer_res = InferTracker()
        m, tracker = models.get_tracked(VectorTextModel.id, infer_res)

        on_processed = tracker('infer')
        used_tokens, vc = m.encode(get_input(request))
        on_processed(tokens=used_tokens)
        return {
            '_usages': infer_res.usages,
            'embeddings': vc.numpy().tolist() if isinstance(vc, Tensor) else vc.astype(np.float32).tolist(),
        }

    @app.route(f'/{VectorTextModel.id}/query', methods=['POST'])
    def vector_text__query():
        return handle_vector_query(VectorTextModel.id)

    @app.route(f'/{VectorCodeModel.id}', methods=['POST'])
    def vector_code():
        infer_res = InferTracker()
        m, tracker = models.get_tracked(VectorCodeModel.id, infer_res)

        on_processed = tracker('infer')
        used_tokens, vc = m.encode(get_input(request))
        on_processed(tokens=used_tokens)
        return {
            '_usages': infer_res.usages,
            'embeddings': vc.numpy().tolist() if isinstance(vc, Tensor) else vc.astype(np.float32).tolist(),
        }

    @app.route(f'/{VectorCodeModel.id}/query', methods=['POST'])
    def vector_code__query():
        return handle_vector_query(VectorCodeModel.id)

    @app.route(f'/{VectorImageModel.id}', methods=['POST'])
    def vector_image():
        infer_res = InferTracker()
        m, tracker = models.get_tracked(VectorImageModel.id, infer_res)

        on_processed = tracker('infer')
        used_tokens, vc = m.encode(get_input(request))
        on_processed(tokens=used_tokens)
        return {
            '_usages': infer_res.usages,
            'embeddings': vc.numpy().tolist() if isinstance(vc, Tensor) else vc.astype(np.float32).tolist(),
        }
