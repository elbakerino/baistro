import json
import threading
import yaml
import os
from typing import List, Tuple, Union

from baistro.config.config import SENTENCE_TRANSFORMERS_HOME, AppConfig
from baistro.model_control.infer_result import InferTracker, ModelTracker
from baistro.model_control.model_base import ModelBase
from baistro.model_control.st_model import SentenceTransformerModelBase
from baistro.models.stanza_models import define_stanza_model, StanzaMultiModel
from baistro.models.vector_code import VectorCodeModel
from baistro.models.vector_image import VectorImageModel
from baistro.models.vector_text import VectorTextModel


def load_models(names: List[str]):
    for filename in names:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                if filename.endswith(('.yaml', '.yml')):
                    return yaml.safe_load(f)
                else:  # .json
                    return json.load(f)

    return None


class Models(object):
    def __init__(self):
        model_classes: List[type(ModelBase)] = [
            StanzaMultiModel,
            VectorCodeModel,
            VectorTextModel,
            VectorImageModel,
        ]
        self.model_classes = {model.id: model for model in model_classes}
        self.model_instances = {}
        self.models_preload = []
        self.lock = threading.Lock()

        # loading models defined in `config.(yml|yaml|json)` in cwd
        # todo: support base and default configs, which allow overwriting models
        config = load_models(AppConfig.MODEL_CONFIG_FILE or ['config.yaml', 'config.yml', 'config.json'])

        if config and 'models' in config:
            for model_config in config['models']:
                model_id = model_config['id']
                model_type = model_config.get('type', 'sentence_transformer')
                model_opts = model_config.get('opts', {})

                if model_id not in self.model_classes:
                    if model_type == 'stanza':
                        self.model_classes[model_id] = define_stanza_model(
                            model_opts['lang'],
                            model_opts.get('tasks'),
                            model_opts.get('processors'),
                        )
                        continue

                    if model_type and model_type != 'sentence_transformer':
                        raise ValueError(f"Unknown model type '{model_type}' for model ID '{model_id}'.")

                    if model_config.get('preload'):
                        # todo: support preload for any model type
                        self.models_preload.append(model_id)

                    # todo: this only works for dynamic vector models
                    # factory function for dynamic class; python last binding
                    def make():
                        class CustomVectorModel(ModelBase):
                            id = model_id

                            name = model_opts['name']
                            description = model_opts.get('description')
                            url = model_opts.get('url', "hugging")
                            folder = f'{SENTENCE_TRANSFORMERS_HOME}/{name.replace("/", "_")}'
                            tasks = ['vector']
                            modality = model_opts.get('modality', ['text'])
                            features = model_opts.get('features', [])

                            _model = None

                            def __init__(self):
                                # note: necessary in init for correct `load` tracking
                                self.model.transformer.to('cpu')

                            @property
                            def model(self):
                                if not self.__class__._model:
                                    self.__class__._model = SentenceTransformerModelBase(self.__class__.folder, local_files_only=True)
                                return self.__class__._model

                            def encode_with_stats(self, sentences: Union[List[str], str], **kwargs):
                                return self.model.encode_with_stats(sentences, **kwargs)

                            def encode(self, sentences: Union[List[str], str], **kwargs):
                                convert_to_tensor = kwargs.pop('convert_to_tensor', True)
                                [tokens, embeddings] = self.model.encode_with_stats(sentences, convert_to_tensor=convert_to_tensor, **kwargs)
                                return embeddings

                            @staticmethod
                            def download():
                                m = CustomVectorModel
                                SentenceTransformerModelBase(m.name).save(m.folder)

                        self.model_classes[model_id] = CustomVectorModel

                    make()
                else:
                    raise ValueError(f"Model ID '{model_id}' conflicts with a pre-defined model.")

    def list(self) -> List[type(ModelBase)]:
        return [m for m in self.model_classes.values()]

    def has(self, model_id) -> bool:
        return model_id in self.model_classes

    def get_type(self, model_id) -> type(ModelBase):
        if not self.has(model_id):
            raise ValueError(f'can not get type for unknown model {model_id}')
        return self.model_classes[model_id]

    def get(self, model_id: str) -> ModelBase:
        if model_id not in self.model_instances:
            self.model_instances[model_id] = self.get_type(model_id)()
        return self.model_instances[model_id]

    def get_tracked(self, model_id: str, infer_res: InferTracker) -> Tuple[ModelBase, ModelTracker]:
        tracker = infer_res.tracker(model_id)
        if model_id not in self.model_instances:
            with self.lock:
                if model_id not in self.model_instances:
                    on_loaded = tracker('load')
                    self.model_instances[model_id] = self.get_type(model_id)()
                    on_loaded()
        return self.model_instances[model_id], tracker


models = Models()
