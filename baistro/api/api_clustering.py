from typing import List
import numpy as np
from flask import Flask, request

from sklearn.cluster import AgglomerativeClustering

from baistro._boot import Services
from baistro.model_control.infer_result import InferTracker
from baistro.model_control.models import models
from baistro.models.vector_text import VectorTextModel


def api_clustering(app: Flask, s: Services):
    @app.route(f'/sentence-clusters', methods=['POST'])
    def sentence_clusters():
        infer_res = InferTracker()
        m, tracker = models.get_tracked(VectorTextModel.id, infer_res)

        corpus = request.json['input']
        if not isinstance(corpus, List):
            return {'error': 'input must be array of sentences'}, 400

        on_processed = tracker('infer')
        used_tokens, corpus_embeddings = m.encode(corpus, batch_size=64)
        on_processed(tokens=used_tokens)

        on_computed = tracker('compute')
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        clustering_model = AgglomerativeClustering(
            n_clusters=None,  # compute_distances=True,
            # distance_threshold=1.5, linkage='ward', metric='euclidean',
            # distance_threshold=1.8, linkage='average', metric='l2',
            distance_threshold=0.6, linkage='average', metric='cosine',
        )
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id not in clustered_sentences:
                clustered_sentences[cluster_id] = []

            clustered_sentences[cluster_id].append([corpus[sentence_id]])

        on_computed()

        return {
            '_usages': infer_res.usages,
            'outcome': {
                'clusters': [cluster for i, cluster in clustered_sentences.items()],
            },
        }
