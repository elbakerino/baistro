from typing import List
from apiflask import APIFlask
from flask import request
from stanza import Document
from stanza.models.common.doc import Sentence
from baistro._boot import Services
from baistro.model_control.infer_result import InferTracker
from baistro.model_control.stanza_model import stanza_model, SIMPLE_PROCESSORS


def api_stanza(app: APIFlask, s: Services):
    @app.route(f'/locale-ident', methods=['POST'])
    @app.doc(tags=[f'NLP'])
    def locale_ident():
        infer_res = InferTracker()
        tracker = infer_res.tracker('stanza-lang_id')

        options = request.json.get('options', {})
        input = request.json['input']

        possible_locales: List[str] = options.get('possible_locales', None)
        clean_text: bool = options.get('clean_text', False)
        if possible_locales:
            possible_locales.sort()

        on_loaded = tracker('load')
        pipe = stanza_model.pipeline(
            locale="multilingual",
            langid_lang_subset=possible_locales,
            langid_clean_text=clean_text,
        )
        on_loaded()

        on_processed = tracker('infer')
        doc: Document = pipe(input, 'langid')
        on_processed(tokens=doc.num_tokens)
        return {
            '_usages': infer_res.usages,
            'outcome': {'locale': doc.lang},
        }

    @app.route(f'/sentence-segments', methods=['POST'])
    @app.doc(tags=[f'NLP'])
    def sentence_segments():
        infer_res = InferTracker()
        tracker = infer_res.tracker('stanza-sentence-segments')

        attributes = request.json.get('attributes', {})
        options = request.json.get('options', {})
        input = request.json['input']

        locale = attributes.get('locale', 'en')
        no_ssplit = options.get('no_ssplit', True)

        on_loaded = tracker('load')
        pipe = stanza_model.pipeline(
            locale=locale,
            tokenize_no_ssplit=no_ssplit,
        )
        on_loaded()

        on_processed = tracker('infer')
        doc: Document = pipe(input, processors='tokenize')
        on_processed(tokens=doc.num_tokens)

        sentence_pieces = []
        for sentence in doc.sentences:
            sentence: Sentence = sentence
            sentence_pieces.append(sentence.text)
        return {
            '_usages': infer_res.usages,
            'outcome': {'sentence_pieces': sentence_pieces},
        }

    @app.route(f'/sentence-classifications', methods=['POST'])
    @app.doc(tags=[f'NLP'])
    def sentence_classifications():
        infer_res = InferTracker()
        tracker = infer_res.tracker('stanza-sentence-classifications')

        attributes = request.json.get('attributes', {})
        options = request.json.get('options', {})
        input = request.json['input']

        locale = attributes.get('locale', 'en')
        no_ssplit = options.get('no_ssplit', True)
        processors_id = options.get('processors', 'lemma')
        processors = ','.join(processors_id) if isinstance(processors_id, List) else (
            SIMPLE_PROCESSORS[processors_id] if processors_id in SIMPLE_PROCESSORS else processors_id
        )

        on_loaded = tracker('load')
        pipe = stanza_model.pipeline(
            locale=locale,
            # if `tokenize_no_ssplit=True` it uses two-newlines to separate sentences,
            # which is important for headline, list-item and other incomplete-sentence to correctly get their struct.
            tokenize_no_ssplit=no_ssplit,
            # tokenize_pretokenized=not isinstance(input, str), # todo: not in caching yet / not needed anymore, or?
        )
        on_loaded()

        on_processed = tracker('infer')
        doc: Document = pipe(
            input if isinstance(input, str) else '\n'.join([' '.join([t['text'] for t in sent['tokens']]) for sent in input]),
            processors=processors,
        )
        on_processed(tokens=doc.num_tokens)

        sentences = []
        for sentence in doc.sentences:
            sentence: Sentence = sentence
            result = {
                # 'id': sentence.sent_id,
            }
            if options.get('include_sentence_entities') and sentence.entities:
                # note: in contenk-annotation, these can not be edited
                result['entities'] = []
                for entity in sentence.entities:
                    result['entities'].append(entity.to_dict())

            if sentence.sentiment is not None:
                sentiments = {
                    0: 'negative',
                    1: 'neutral',
                    2: 'positive',
                }
                result['sentiment'] = sentence.sentiment
                result['sentiment_name'] = sentiments[sentence.sentiment]

            # for dependency in sentence.dependencies:
            #     print('dependency', type(dependency), dependency)

            result['mwt_tokens'] = []
            result['tokens'] = []
            ignore_ids = {}
            for token in sentence.tokens:
                # note: this is all words incl. their POS/NER/lemma etc.
                token_dicts = token.to_dict()
                for token_dict in token_dicts:
                    if 'start_char' in token_dict:
                        token_dict.pop('start_char')
                    if 'end_char' in token_dict:
                        token_dict.pop('end_char')

                    # splitting up MWTs, only keeping the org. token in the list
                    if isinstance(token_dict['id'], list) or isinstance(token_dict['id'], tuple):
                        for tid in token_dict['id']:
                            ignore_ids[tid] = token_dict['id']
                        result['tokens'].append(token_dict)
                    else:
                        if token_dict['id'] in ignore_ids:
                            token_dict['mwt'] = ignore_ids.pop(token_dict['id'])
                            result['mwt_tokens'].append(token_dict)
                        else:
                            result['tokens'].append(token_dict)

            if not result['mwt_tokens']:
                result.pop('mwt_tokens')

            if options.get('include_plain_words'):
                result['plain_words'] = []
                for word in sentence.words:
                    # note: this only includes the very basic info,
                    #       but not POS, lemma or ner of the words
                    result['plain_words'].append(word.to_dict())

            if options.get('include_comments'):
                result['comments'] = sentence.comments

            # print('constituency', type(sentence.constituency), sentence.constituency)
            sentences.append(result)

        result_full = {
            'sentences': sentences,
        }
        if options.get('include_doc_entities'):
            result_full['entities'] = [entity.to_dict() for entity in doc.entities]

        return {
            '_usages': infer_res.usages,
            'outcome': result_full,
        }
