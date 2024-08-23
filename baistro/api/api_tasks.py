from PIL import Image
from apiflask import fields, APIFlask
from flask import request
from baistro.api.schemas import OutcomeTextPairsResponse, OutcomeWithScoresResponse
from baistro._boot import Services
from baistro.helper.qag_parser import qag_split_pairs
from baistro.model_control.infer_result import InferTracker
from baistro.model_control.models import models
from baistro.models.dit_base import DitBaseModel
from baistro.models.dit_large import DitLargeModel
from baistro.models.donut_docvqa import DonutDocvqaModel
from baistro.models.donut_to_data import DonutToDataModel
from baistro.models.donut_to_text import DonutToTextModel
from baistro.models.qag_en_base import QagEnBaseModel
from baistro.models.qnli_en_base import QnliEnBaseModel
from baistro.models.qa_en_large import QaEnLargeModel
from baistro.models.qa_en_base import QaEnBaseModel


def api_tasks(app: APIFlask, s: Services):
    @app.route(f'/{QagEnBaseModel.id}', methods=['POST'])
    @app.input({'input': fields.String()}, schema_name=f'Input{QagEnBaseModel.id}')
    @app.output(OutcomeTextPairsResponse())
    @app.doc(tags=[f'Task: {task}' for task in QagEnBaseModel.tasks])
    def qag_en_base(json_data):
        infer_res = InferTracker()
        m, tracker = models.get_tracked(QagEnBaseModel.id, infer_res)

        input = json_data['input']

        on_processed = tracker('infer')
        used_tokens, result = m.generate('generate question and answer: ' + input)
        on_processed(tokens=used_tokens)
        pairs = qag_split_pairs(result)
        return {
            '_usages': infer_res.usages,
            'outcome': pairs,
        }

    @app.route(f'/{QnliEnBaseModel.id}', methods=['POST'])
    # todo: tuple seems to be not supported? https://github.com/marshmallow-code/apispec/issues/399
    # @app.input({'input': fields.List(fields.Tuple((fields.String(), fields.String())))})
    @app.input({'input': fields.List(fields.List(fields.String()))}, schema_name=f'Input{QnliEnBaseModel.id}')
    @app.output(OutcomeWithScoresResponse())
    @app.doc(tags=[f'Task: {task}' for task in QnliEnBaseModel.tasks])
    def qnli_en_base(json_data):
        infer_res = InferTracker()
        m, tracker = models.get_tracked(QnliEnBaseModel.id, infer_res)

        input = json_data['input']

        on_processed = tracker('infer')
        used_tokens, result = m.generate(input)
        on_processed(tokens=used_tokens)
        return {
            '_usages': infer_res.usages,
            'outcome': result,
        }

    @app.route(f'/{QaEnBaseModel.id}', methods=['POST'])
    @app.doc(tags=[f'Task: {task}' for task in QaEnBaseModel.tasks])
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
    @app.doc(tags=[f'Task: {task}' for task in QaEnLargeModel.tasks])
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
    @app.doc(tags=[f'Task: {task}' for task in DonutDocvqaModel.tasks])
    def donut_docvqa():
        return handle_vqa(DonutDocvqaModel.id)

    @app.route(f'/{DonutToDataModel.id}', methods=['POST'])
    @app.doc(tags=[f'Task: {task}' for task in DonutToDataModel.tasks])
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
    @app.doc(tags=[f'Task: {task}' for task in DonutToTextModel.tasks])
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
    @app.doc(tags=[f'Task: {task}' for task in DitBaseModel.tasks])
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
    @app.doc(tags=[f'Task: {task}' for task in DitLargeModel.tasks])
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
