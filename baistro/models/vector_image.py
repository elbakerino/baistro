from typing import Union, List
from PIL.Image import Image
from baistro.model_control.model_base import ModelBase
from baistro.model_control.st_model import SentenceTransformerModelBase
from baistro.config.config import AppConfig, SENTENCE_TRANSFORMERS_HOME


class VectorImageModel(ModelBase):
    id = 'vector-image'
    if 'VECTOR_IMAGE' in AppConfig.MODELS:
        name = AppConfig.MODELS['VECTOR_IMAGE']
    else:
        name = "sentence-transformers/clip-ViT-B-32"
    url = "hugging"
    folder = f'{SENTENCE_TRANSFORMERS_HOME}/{name.replace("/", "_")}'
    # folder = f'{os.getenv("SENTENCE_TRANSFORMERS_HOME")}/{name.replace("/", "_")}'
    tasks = ['vector-text', 'vector-image']

    _model = None

    def __init__(self):
        # note: necessary in init for correct `load` tracking
        self.model.transformer.to('cpu')

    @property
    def model(self):
        if not VectorImageModel._model:
            VectorImageModel._model = SentenceTransformerModelBase(VectorImageModel.folder, local_files_only=True)
        return VectorImageModel._model

    def encode(self, sentences: Union[List[str], str, Image]):
        return self.model.encode_with_stats(sentences, convert_to_tensor=True)

    @staticmethod
    def download():
        m = VectorImageModel
        SentenceTransformerModelBase(m.name).save(m.folder)
