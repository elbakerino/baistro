import threading

from stanza import Pipeline

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
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.pipelines = {}
        self.lock = threading.Lock()

    def pipeline(
        self,
        locale: str,
        **kwargs,
    ) -> Pipeline:
        cache_id = f'{locale}'
        if 'langid_lang_subset' in kwargs and kwargs['langid_lang_subset']:
            cache_id += '|' + ','.join(sorted(kwargs['langid_lang_subset']))
        if 'langid_clean_text' in kwargs and kwargs['langid_clean_text']:
            cache_id += '|' + ('clean' if kwargs['langid_clean_text'] else 'keep')
        if 'tokenize_no_ssplit' in kwargs and kwargs['tokenize_no_ssplit']:
            cache_id += '|no_ssplit'
        if 'tokenize_pretokenized' in kwargs and kwargs['tokenize_pretokenized']:
            cache_id += '|pretokenized'
        if 'processors' in kwargs and kwargs['processors']:
            cache_id += '|' + kwargs['processors']

        if cache_id not in self.pipelines:
            with self.lock:
                if cache_id not in self.pipelines:
                    self.pipelines[cache_id] = Pipeline(
                        locale,
                        dir=self.model_dir,
                        download_method=None,
                        verbose=False,
                        **kwargs,
                    )

        return self.pipelines[cache_id]


stanza_model = StanzaModelCache(AppConfig.MODEL_DIR_STANZA)
