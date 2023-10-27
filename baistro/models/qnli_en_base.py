from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from baistro.config.config import AppConfig
from baistro.model_control.model_base import ModelBase


class QnliEnBaseModel(ModelBase):
    name = "cross-encoder/qnli-electra-base"
    url = "hugging"
    folder = f'{AppConfig.MODEL_DIR}/model-{name.replace("/", "_").lower()}'
    id = 'qnli-en-base'
    tasks = ['qnli']
    locale = ['en']

    _model = None
    _tokenizer = None

    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

    @property
    def model(self):
        if not QnliEnBaseModel._model:
            QnliEnBaseModel._model = AutoModelForSequenceClassification.from_pretrained(self.folder)
        return QnliEnBaseModel._model

    @property
    def tokenizer(self):
        if not QnliEnBaseModel._tokenizer:
            QnliEnBaseModel._tokenizer = AutoTokenizer.from_pretrained(self.folder)
        return QnliEnBaseModel._tokenizer

    def generate(self, pairs: List[Tuple[str, str]]):
        questions = [q for (q, _) in pairs]
        answers = [a for (_, a) in pairs]
        features = self.tokenizer(
            questions, answers,
            padding=True, truncation=True, return_tensors="pt",
        )
        tokens = sum(len(inp_ids) for inp_ids in features['input_ids'])

        self.model.eval()
        with torch.no_grad():
            pred_scores = torch.nn.functional.sigmoid(self.model(**features).logits)
            scores = []
            for i, score in enumerate(pred_scores):
                scores.append(float(score))
            return tokens, scores

    @staticmethod
    def download():
        m = QnliEnBaseModel
        tokenizer = AutoTokenizer.from_pretrained(m.name)
        model = AutoModelForSequenceClassification.from_pretrained(m.name)
        tokenizer.save_pretrained(m.folder)
        model.save_pretrained(m.folder)
