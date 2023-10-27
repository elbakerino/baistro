import os
from typing import Union, List
from sentence_transformers import SentenceTransformer
from baistro.config.config import AppConfig
from baistro.model_control.model_base import ModelBase
from baistro.model_control.st_model import SentenceTransformerModelBase


class VectorTextModel(ModelBase):
    id = 'vector-text'

    if 'VECTOR_TEXT' in AppConfig.MODELS:
        name = AppConfig.MODELS['VECTOR_TEXT']
    else:
        name = "sentence-transformers/all-distilroberta-v1"
    url = "hugging"
    folder = f'{os.getenv("SENTENCE_TRANSFORMERS_HOME")}/{name.replace("/", "_")}'
    tasks = ['vector-text']

    _model = None

    def __init__(self):
        self.model.to('cpu')

    @property
    def model(self):
        if not VectorTextModel._model:
            VectorTextModel._model = SentenceTransformerModelBase(VectorTextModel.folder)
        return VectorTextModel._model

    def encode(self, sentences: Union[List[str], str], **kwargs):
        return self.model.encode_with_stats(sentences, convert_to_tensor=True, **kwargs)

    @staticmethod
    def download():
        m = VectorTextModel
        SentenceTransformer(m.name)
