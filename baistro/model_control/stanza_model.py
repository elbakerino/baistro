import threading

import stanza

from baistro.config.config import AppConfig

SIMPLE_PROCESSORS = {
    'mwt': 'tokenize',
    'pos': 'tokenize,mwt,pos',
    'lemma': 'tokenize,mwt,pos,lemma',
    'depparse': 'tokenize,mwt,pos,lemma,depparse',
    'ner': 'tokenize,mwt,ner',
    'sentiment': 'tokenize,mwt,sentiment',
    'constituency': 'tokenize,mwt,pos,constituency',
    '_most': 'tokenize,mwt,pos,lemma,ner',
    '_all': 'tokenize,mwt,pos,lemma,ner,sentiment,constituency,depparse',
}


class StanzaModelCache(object):
    processor_maps = {
        'multilingual': 'langid',
        'en': 'tokenize,pos,mwt,ner,lemma,depparse,sentiment,constituency',
        'de': 'tokenize,pos,mwt,ner,lemma,depparse,sentiment',
    }

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.pipelines = {}
        self.lock = threading.Lock()

    def pipeline(
        self,
        locale: str,
        **kwargs,
    ) -> stanza.Pipeline:
        cache_id = f'{locale}'
        if 'langid_lang_subset' in kwargs and kwargs['langid_lang_subset']:
            cache_id += '|' + ','.join(sorted(kwargs['langid_lang_subset']))
        if 'langid_clean_text' in kwargs and kwargs['langid_clean_text']:
            cache_id += '|' + ('clean' if kwargs['langid_clean_text'] else 'keep')
        if 'tokenize_no_ssplit' in kwargs and kwargs['tokenize_no_ssplit']:
            cache_id += '|no_ssplit'

        if cache_id not in self.pipelines:
            with self.lock:
                if cache_id not in self.pipelines:
                    self.pipelines[cache_id] = stanza.Pipeline(
                        locale,
                        dir=self.model_dir,
                        download_method=None,
                        verbose=False,
                        **kwargs,
                    )

        return self.pipelines[cache_id]


stanza_model = StanzaModelCache(AppConfig.MODEL_DIR + '/_stanza')
