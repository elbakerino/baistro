import os
from typing import Union, List
import stanza
from baistro.model_control.model_base import ModelBase
from baistro.model_control.stanza_model import StanzaModelCache


def stanza_download(locale: str):
    stanza.download(
        locale,
        processors=StanzaModelCache.processor_maps[locale],
        model_dir=os.getenv("STANZA_RESOURCES_DIR"),
        logging_level='INFO',
    )


def define_stanza_model(model_locale: str, model_tasks: Union[List[str], None] = None):
    # only used for stats/download, as rest is handled by `StanzaModelCache`
    class StanzaModel(ModelBase):
        name = 'stanza-' + model_locale
        url = "https://stanfordnlp.github.io/stanza/"
        folder = f'{os.getenv("STANZA_RESOURCES_DIR")}/' + model_locale
        id = 'stanza-' + model_locale
        locale = [model_locale]
        tasks = model_tasks if model_tasks else [
            'sequence-classification',
            'token-classification',
            'pos', 'ner',
        ]

        @staticmethod
        def download():
            stanza_download(model_locale)

    return StanzaModel


StanzaMultiModel = define_stanza_model('multilingual', ['locale-identification'])
# todo: find a way to dynamize this
StanzaBgModel = define_stanza_model('bg')
StanzaDaModel = define_stanza_model('da')
StanzaDeModel = define_stanza_model('de', [
    'sequence-classification',
    'token-classification',
    'pos', 'ner',
    'sentiment',
])
StanzaEnModel = define_stanza_model('en', [
    'sequence-classification',
    'token-classification',
    'pos', 'ner',
    'sentiment',
])
StanzaEsModel = define_stanza_model('es')
StanzaFiModel = define_stanza_model('fi')
StanzaFrModel = define_stanza_model('fr')
StanzaHuModel = define_stanza_model('hu')
StanzaItModel = define_stanza_model('it')
StanzaJaModel = define_stanza_model('ja')
StanzaNlModel = define_stanza_model('nl')
StanzaNnModel = define_stanza_model('nn')
StanzaPlModel = define_stanza_model('pl')
StanzaSvModel = define_stanza_model('sv')
