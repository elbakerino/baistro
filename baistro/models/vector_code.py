from typing import Union, List
from baistro.model_control.model_base import ModelBase
from baistro.model_control.st_model import SentenceTransformerModelBase
from baistro.config.config import AppConfig, SENTENCE_TRANSFORMERS_HOME


class VectorCodeModel(ModelBase):
    id = 'vector-code'
    if 'VECTOR_CODE' in AppConfig.MODELS:
        name = AppConfig.MODELS['VECTOR_CODE']
    else:
        name = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
    url = "hugging"
    folder = f'{SENTENCE_TRANSFORMERS_HOME}/{name.replace("/", "_")}'
    # folder = f'{os.getenv("SENTENCE_TRANSFORMERS_HOME")}/{name.replace("/", "_")}'
    tasks = ['vector-text', 'vector-code']
    locale = ['en']

    _model = None

    def __init__(self):
        # note: necessary in init for correct `load` tracking
        self.model.transformer.to('cpu')

    @property
    def model(self):
        if not VectorCodeModel._model:
            VectorCodeModel._model = SentenceTransformerModelBase(VectorCodeModel.folder, local_files_only=True)
        return VectorCodeModel._model

    def encode(self, sentences: Union[List[str], str]):
        return self.model.encode_with_stats(sentences, convert_to_tensor=True)

    @staticmethod
    def download():
        m = VectorCodeModel
        SentenceTransformerModelBase(m.name).save(m.folder)
