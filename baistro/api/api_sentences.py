from typing import List
import numpy as np
from flask import Flask, request

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from baistro._boot import Services
from baistro.model_control.infer_result import InferTracker


def api_sentences(app: Flask, s: Services):
    @app.route(f'/sentence-word-occurrences', methods=['POST'])
    def sentence_word_occurrences():
        infer_res = InferTracker()
        tracker = infer_res.tracker('scikit')

        corpus = request.json['input']
        if not isinstance(corpus, List) and not isinstance(corpus, str):
            return {'error': 'input must be array|str'}, 400

        on_computed = tracker('compute')

        vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            # token_pattern=r'\b\w+\b', min_df=1,
            stop_words=[
                'this', 'that', 'for', 'a', 'an', 'is', 'it', 'the', 'or', 'and',
                'if', 'of', 'on', 'by', 'to', 'can', 'also',
            ],
            # max_features=10,
        )
        x: csr_matrix = vectorizer.fit_transform(corpus)

        feature_names = vectorizer.get_feature_names_out()
        feature_counts = x.sum(axis=0)
        variance = feature_counts.var()
        sparsity = 1.0 - (x.count_nonzero() / (x.shape[0] * x.shape[1]))

        feature_counts = np.squeeze(np.asarray(feature_counts))

        features_with_counts = list(zip(feature_names, feature_counts))
        features_with_counts.sort(key=lambda x: x[1], reverse=True)

        word_counts = []
        for feature, count in features_with_counts:
            word_counts.append([feature, int(count)])

        on_computed()

        return {
            '_usages': infer_res.usages,
            'outcome': {
                'variance': variance,
                'sparsity': sparsity,
                'words': word_counts,
            },
        }
