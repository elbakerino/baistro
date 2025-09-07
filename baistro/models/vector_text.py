from typing import Union, List
from baistro.config.config import AppConfig, SENTENCE_TRANSFORMERS_HOME
from baistro.model_control.model_base import ModelBase
from baistro.model_control.st_model import SentenceTransformerModelBase


class VectorTextModel(ModelBase):
    id = 'vector-text'

    if 'VECTOR_TEXT' in AppConfig.MODELS:
        name = AppConfig.MODELS['VECTOR_TEXT']
    else:
        name = "sentence-transformers/all-distilroberta-v1"
    url = "hugging"
    folder = f'{SENTENCE_TRANSFORMERS_HOME}/{name.replace("/", "_")}'
    # folder = f'{os.getenv("SENTENCE_TRANSFORMERS_HOME")}/{name.replace("/", "_")}'
    tasks = ['vector', 'vector-text']
    modality = ['text']
    features = ['search']

    _model = None

    def __init__(self):
        # note: necessary in init for correct `load` tracking
        self.model.transformer.to('cpu')

    @property
    def model(self):
        if not VectorTextModel._model:
            VectorTextModel._model = SentenceTransformerModelBase(VectorTextModel.folder, local_files_only=True)
        return VectorTextModel._model

    def encode_with_stats(self, sentences: Union[List[str], str], **kwargs):
        return self.model.encode_with_stats(sentences, **kwargs)

    def encode(self, sentences: Union[List[str], str], **kwargs):
        convert_to_tensor = kwargs.pop('convert_to_tensor', True)
        [tokens, embeddings] = self.model.encode_with_stats(sentences, convert_to_tensor=convert_to_tensor, **kwargs)
        return embeddings

    @staticmethod
    def download():
        m = VectorTextModel
        SentenceTransformerModelBase(m.name).save(m.folder)
