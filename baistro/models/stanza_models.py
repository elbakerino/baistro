import os
from typing import Union, List
import stanza
from baistro.model_control.model_base import ModelBase

processor_maps = {
    'multilingual': 'langid',
    'en': 'tokenize,pos,mwt,ner,lemma,depparse,sentiment,constituency',
    'de': 'tokenize,pos,mwt,ner,lemma,depparse,sentiment,constituency',
}


def stanza_download(locale: str, processors=None):
    stanza.download(
        locale,
        processors=processor_maps.get(locale) if processors is None else processors,
        model_dir=os.getenv("STANZA_RESOURCES_DIR"),
        logging_level='INFO',
    )


def define_stanza_model(
    model_locale: str,
    model_tasks: Union[List[str], None] = None,
    processors=None,
):
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
            stanza_download(model_locale, processors)

    return StanzaModel


StanzaMultiModel = define_stanza_model('multilingual', ['locale-identification'])
