import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from baistro.config.config import AppConfig
from baistro.model_control.model_base import ModelBase


class QaEnBaseModel(ModelBase):
    name = "deepset/deberta-v3-base-squad2"
    url = "hugging"
    folder = f'{AppConfig.MODEL_DIR}/model-{name.replace("/", "_").lower()}'
    id = 'qa-en-base'
    tasks = ['qa']
    locale = ['en']

    _model = None
    _tokenizer = None

    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

    @property
    def model(self):
        if not QaEnBaseModel._model:
            QaEnBaseModel._model = AutoModelForQuestionAnswering.from_pretrained(self.folder)
        return QaEnBaseModel._model

    @property
    def tokenizer(self):
        if not QaEnBaseModel._tokenizer:
            QaEnBaseModel._tokenizer = AutoTokenizer.from_pretrained(self.folder)
        return QaEnBaseModel._tokenizer

    def generate(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors="pt")
        tokens = sum(len(inp_ids) for inp_ids in inputs['input_ids'])

        with torch.no_grad():
            outputs = self.model(**inputs)
            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax()
            predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]

            answer = self.tokenizer.decode(predict_answer_tokens)

            return tokens, answer

    @staticmethod
    def download():
        m = QaEnBaseModel
        tokenizer = AutoTokenizer.from_pretrained(m.name)
        model = AutoModelForQuestionAnswering.from_pretrained(m.name)
        tokenizer.save_pretrained(m.folder)
        model.save_pretrained(m.folder)
