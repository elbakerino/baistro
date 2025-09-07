from typing import List, Union
import numpy as np
from apiflask import APIFlask
from marshmallow import validates, ValidationError
from marshmallow.validate import OneOf
from sklearn.cluster import AgglomerativeClustering, OPTICS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
from stanza import Document
from stanza.models.common.doc import Sentence

from baistro._boot import Services
from baistro.api.clustering_plots import make_cluster_plots, _plot_options, _plotter_fn
from baistro.model_control.infer_result import InferTracker
from baistro.model_control.model_base import ModelBase
from baistro.model_control.models import models
from baistro.model_control.stanza_model import stanza_model
from apiflask import fields, Schema, HTTPError
from baistro.api.schemas import InferBaseResponse, StringOrList


class SentenceClusterRequestThresholds(Schema):
    predefined_label = fields.Float(
        metadata={
            'default': 0.32,
            'description': 'Similarity threshold for assigning a predefined label to a cluster. Only used if `predefined_labels` are provided.'
        }
    )
    predefined_keyword = fields.Float(
        metadata={
            'default': 0.32,
            'description': 'Minimum similarity score for a `predefined_keyword` to be considered as a candidate for the keywords list.'
        }
    )
    dynamic_label = fields.Float(
        metadata={
            'default': 0.0,
            'description': 'Minimum score for a dynamically generated candidate to be considered for the cluster label. A value of `0.0` (default) means any top-scoring dynamic candidate can be used if no predefined label meets its threshold.'
        }
    )
    dynamic_keyword = fields.Float(
        metadata={
            'default': 0.0,
            'description': 'Minimum score for a dynamically generated candidate to be included in the final keywords list. A value of `0.0` (default) means any top-scoring dynamic candidates can be included.'
        }
    )


class SentenceClusterRequestKeywordsOptions(Schema):
    tokenization = fields.String(
        metadata={
            'default': 'tfidf',
            'description': '''The method used to identify candidate keywords from the text within each cluster.

- **`tfidf`**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to find words or phrases that are important within a cluster relative to the entire document set. It's excellent for finding distinctive topics.

- **`count`**: Uses simple term frequency. It identifies the most common words or phrases within a cluster. This is faster but less nuanced than TF-IDF.

- **`stanza`**: Uses advanced NLP processing to extract noun phrases and named entities. This method often yields higher-quality, more meaningful keywords but is computationally more intensive.

Start with `tfidf` for a good balance of quality and performance. Use `stanza` for the highest quality keywords if performance is not a major concern. Use `count` for maximum speed.'''
        },
        validate=OneOf(['count', 'tfidf', 'stanza']),
    )
    ngram_range = fields.List(
        fields.Integer(),
        allow_none=True,
        metadata={
            'default': [1, 3],
            'description': '''The range of n-grams to consider as keyword candidates (e.g., `[1, 3]` means unigrams, bigrams, and trigrams).

A range of `[1, 3]` is effective for capturing single words and short, meaningful phrases. A smaller range like `[1, 1]` will only find single-word keywords. This setting is only applicable for `count` and `tfidf` tokenization.'''
        },

    )
    stop_words = StringOrList(
        metadata={
            'default': None,
            'description': '''A list of words to ignore during keyword extraction.

You can provide a custom list of words, use the built-in `'english'` list, or set to `None` to use a small, default list of common English words. Providing a custom list tailored to your domain can significantly improve keyword quality by filtering out noise. This setting is only applicable for `count` and `tfidf` tokenization.'''
        },
        allow_none=True,
    )
    max_candidates = fields.Integer(
        metadata={
            'default': None,
            'description': '''The maximum number of keyword candidates to generate per cluster when using the `fixed_count` strategy.

This provides a simple, predictable way to cap the number of potential keywords considered, which can be useful for performance. For more dynamic and context-aware control, use other strategies like `proportional` or `adaptive`.'''
        },
        allow_none=True,
    )
    candidate_selection_strategy = fields.String(
        metadata={
            'default': 'proportional',
            'description': '''The strategy for determining how many keyword candidates to generate for each cluster.

- **`fixed_count`**: Uses the static value from `max_candidates`. Simple and predictable.
- **`proportional`**: Scales the number of candidates based on cluster size (token count). Larger clusters generate more candidates. Controlled by `proportional_*` options.
- **`density`**: Scales the number of candidates based on cluster cohesion (average internal distance). Sparser, less-focused clusters generate more candidates to explore potential themes. Controlled by `density_*` options.
- **`logarithmic`**: Scales the number of candidates based on the logarithm of the cluster's token count. This provides more candidates for small clusters than `proportional` but grows slower for large clusters. Controlled by `log_*` options.
- **`adaptive`**: A hybrid strategy that uses the "elbow" or "knee" method on TF-IDF scores to find a natural cutoff point for the most significant candidates. This adapts dynamically to the score distribution within the cluster. Controlled by `tfidf_adaptive_percentile` (used as a fallback). *This strategy is only valid for `tfidf` tokenization.*
'''
        },
        validate=OneOf(['fixed_count', 'proportional', 'density', 'adaptive', 'logarithmic']),
    )
    proportional_ratio = fields.Float(
        metadata={
            'default': 0.8,
            'description': 'For the `proportional` strategy, this is the ratio of the total number of unique terms in a cluster to use as candidates. For example, a value of `0.1` would select the top 10% most frequent terms as candidates.'
        },
    )
    proportional_min = fields.Integer(
        metadata={
            'default': None,
            'description': 'The minimum number of candidates to generate for any cluster. This is a floor for strategies like `proportional`, `logarithmic`, and `adaptive` to ensure even very small clusters have a baseline number of candidates to be evaluated.'
        },
        allow_none=True,
    )
    proportional_max = fields.Integer(
        metadata={
            'default': 1000,
            'description': 'The maximum number of candidates to generate for any cluster. This acts as a global ceiling for strategies like `proportional`, `logarithmic`, and `adaptive` to prevent very large clusters from generating an excessive number of candidates, which can impact performance.'
        },
    )
    proportional_max_local = fields.Integer(
        metadata={
            'default': None,
            'description': 'Overrides `proportional_max` specifically for candidates generated from the `local` source (i.e., from text within a single cluster). If not set, the global `proportional_max` is used. This allows for finer control, for example, allowing more global candidates while restricting local ones.'
        },
        allow_none=True,
    )
    proportional_max_global = fields.Integer(
        metadata={
            'default': None,
            'description': 'Overrides `proportional_max` specifically for candidates generated from the `global` source (i.e., from the entire corpus). If not set, the global `proportional_max` is used. This is useful for tuning the balance between globally relevant and locally specific terms.'
        },
        allow_none=True,
    )
    log_scale_factor = fields.Integer(
        metadata={'default': 100, 'description': 'For the `logarithmic` strategy, this is a scaling factor to adjust the number of candidates. A larger value will increase the number of candidates generated for a given cluster size.'}
    )
    log_base = fields.Float(
        metadata={'default': 10.0, 'description': 'For the `logarithmic` strategy, this is the base of the logarithm used for scaling. A higher base will cause the number of candidates to grow more slowly as cluster size increases.'}
    )
    density_base = fields.Integer(metadata={'default': 100, 'description': 'For the `density` strategy, this is the base number of candidates to generate for a perfectly dense cluster (average internal distance of 0). It sets the minimum for this strategy.'})
    density_scale = fields.Integer(metadata={'default': 400, 'description': 'For the `density` strategy, this is the scaling factor that determines how many *additional* candidates are generated as a cluster becomes less dense (more spread out). A higher value means sparser clusters get significantly more candidates.'})
    density_max_dist = fields.Float(metadata={'default': 0.5, 'description': 'For the `density` strategy, this is the expected maximum average internal distance for a cluster, used to normalize the density calculation. Distances beyond this value are capped, preventing extremely sparse clusters from generating an excessive number of candidates.'})
    tfidf_adaptive_percentile = fields.Integer(
        metadata={
            'default': 90,
            'description': 'For `adaptive` strategy: the percentile (0-100) of TF-IDF scores to use as the cutoff. For example, a value of `90` means only candidates in the top 10% of TF-IDF scores will be considered. Higher values are stricter.'
        },
        validate=lambda n: 0 <= n <= 100,
    )
    top_keywords = fields.Integer(
        metadata={
            'default': 3,
            'description': 'The final number of top keywords to return for each cluster after all candidates have been generated, scored, and ranked. This controls the size of the final `keywords` list in the output.'
        },
    )
    semantic_weight = fields.Float(
        metadata={
            'default': 0.75,
            'description': '''The weight given to semantic similarity when ranking keywords.

A higher value prioritizes keywords that are semantically closest to the cluster's central theme (centroid). A lower value gives more importance to frequency.'''
        },
    )
    freq_weight = fields.Float(
        metadata={
            'default': 0.25,
            'description': '''The weight given to term frequency (or TF-IDF score) when ranking keywords.

A higher value prioritizes keywords that appear more often within the cluster. A lower value gives more importance to semantic relevance.'''
        },
    )
    labels = fields.List(
        fields.String(),
        metadata={
            'description': '''An optional list of predefined labels to assign to clusters.

If provided, the API will compare each cluster's centroid to the embeddings of these labels. The closest label will be assigned if its similarity score is above the `labels_threshold`. This is useful for categorizing clusters into a known set of topics.''',
            'example': ['food', 'movies', 'music', 'business'],
        },
    )
    predefined_labels = fields.Dict(
        keys=fields.String(),
        values=fields.List(fields.String()),
        metadata={
            'description': '''An optional dictionary to "pre-fit" or "fine-tune" the semantic meaning of labels using few-shot examples. This is the primary way to provide predefined labels.

Provide a mapping from a label name to a list of example sentences that define that label's concept. The system will average the embeddings of these examples to create a more robust "concept vector" for each label. This concept vector is then used for matching against clusters, often yielding much better results than just using the label word itself. This is more powerful than `labels` but also more verbose. If a label exists in both `predefined_labels` and `labels`, `predefined_labels` takes precedence.''',
            'example': {
                'technology_innovation': [
                    "The new AI model achieved state-of-the-art results.",
                    "Quantum computing promises to revolutionize the industry.",
                    "Automation and digitalization will continue to reshape industries and daily life."
                ],
                'finance': []
            },
        },
    )
    predefined_keywords = fields.Dict(
        keys=fields.String(),
        values=fields.List(fields.String()),
        metadata={
            'description': '''An optional dictionary to provide a fixed set of candidate keywords with few-shot examples.

Similar to `predefined_labels`, this allows you to define a specific vocabulary of keywords you want to rank for each cluster. The system will create concept vectors for each keyword and use them as the sole candidates if `disable_candidate_generation` is true, or add them to the pool of generated candidates if it's false. This is useful for scoring clusters against a controlled taxonomy.''',
            'example': {
                'schema': ["describes the shape of data", "a contract for apis"],
                'clustering': ["grouping similar items", "unsupervised learning"]
            },
        },
    )
    dynamic_candidate_source = fields.List(
        fields.String(validate=OneOf(['labels', 'keywords'])),
        metadata={
            'default': None,
            'description': '''Specifies which outputs should use dynamically generated candidates (from `local` or `global` sources).
- **`labels`**: Dynamic candidates are used for selecting the main cluster label.
- **`keywords`**: Dynamic candidates are used for generating the list of supporting keywords.

By default (if `null`), dynamic candidates are used for both. To use `predefined_keywords` exclusively for labeling but still discover new supporting keywords from the text, you would set this to `['keywords']`.'''
        },
        allow_none=True,
    )
    force_predefined_for = fields.List(
        fields.String(validate=OneOf(['labels', 'keywords'])),
        metadata={
            'default': None,
            'description': '''Specifies for which outputs the closest predefined candidate should be forcibly used, regardless of its similarity score.
- **`labels`**: Forces the assignment of the closest predefined label to each cluster, even if its similarity is below the threshold.
- **`keywords`**: Forces the top keywords to be selected only from the `predefined_keywords` list.

This is useful when you want to categorize clusters strictly into a known taxonomy. For example, setting `['labels']` ensures every cluster gets a predefined label, while still allowing dynamic keywords to be discovered from the text.
'''
        },
        allow_none=True,
    )
    mmr_lambda = fields.Float(
        metadata={
            'default': 0.0,
            'description': '''Lambda parameter for Maximal Marginal Relevance (MMR) to diversify keywords.

A value of `0.0` disables MMR, selecting keywords based purely on their score (a mix of similarity and frequency). A value closer to `1.0` increases diversity by penalizing keywords that are too similar to already selected ones. A good starting point for diversity is `0.5`.'''
        }
    )
    stanza_processors = fields.String(
        metadata={
            'default': 'tokenize,pos',
            'description': '''A comma-separated list of Stanza processors to use when `tokenization` is `stanza`.

`tokenize,pos` is the minimum required for noun phrase extraction. Add `lemma` for lemmatization (e.g., 'cars' -> 'car') and `ner` for named entity recognition. More processors provide richer data but increase processing time.''',
            'examples': [
                'tokenize,pos',
                'tokenize,pos,lemma',
                'tokenize,pos,lemma,ner',
                'tokenize,pos,ner',
            ]
        },
    )
    lift_weight = fields.Float(
        metadata={
            'default': 0.0,
            'description': '''The weight given to a keyword's "lift" or "specificity" when ranking.

Lift measures how much more frequent a keyword is within its cluster compared to its frequency across the entire document set. A value greater than `0.0` (e.g., `0.25`) will boost the score of keywords that are uniquely concentrated in a cluster and penalize keywords that are common everywhere, leading to more specific and descriptive labels. This is a sophisticated, data-driven way to favor local candidates.'''
        }
    )
    global_penalty_factor = fields.Float(
        metadata={
            'default': 1.0,
            'description': '''A simple multiplicative penalty applied to keywords that are found in the global corpus but not within the local cluster's own text.

A value between `0.0` and `1.0` (e.g., `0.8`) will reduce the score of these "global-only" candidates, favoring terms that are explicitly present in the cluster. A value of `1.0` (default) applies no penalty. This provides a direct, simple way to penalize non-local candidates.'''
        },
        validate=lambda n: 0.0 <= n <= 1.0,
    )
    max_features = fields.Integer(
        metadata={
            'default': None,
            'description': 'Limits the vocabulary size for `count` or `tfidf` tokenization by keeping only the `max_features` most frequent terms across the entire corpus. This can significantly improve performance and reduce noise on very large or diverse datasets by filtering out rare, potentially irrelevant terms.'
        },
        allow_none=True,
    )
    candidate_sources = fields.List(
        fields.String(validate=OneOf(['local', 'global'])),
        allow_none=True,
        metadata={
            'default': ['global', 'local'],
            'description': '''Sources for generating keyword candidates.

- **`local`**: Generates candidates only from the text within each individual cluster. Good for surfacing rare or specific topics.
- **`global`**: Generates candidates from the entire input corpus. Good for finding globally relevant and normalized keywords.

Using both `['global', 'local']` provides the most comprehensive set of candidates but is more computationally expensive. Using only `local` is faster and focuses on cluster-specific terms.'''
        },
    )
    thresholds = fields.Nested(
        SentenceClusterRequestThresholds(),
        metadata={'description': 'A container for various similarity and scoring thresholds.'}
    )
    candidate_pruning_strategy = fields.String(
        metadata={
            'default': None,
            'description': '''Strategy for pruning keyword candidates based on their document frequency. This is applied after initial candidate generation.

- **`min_df`**: Removes candidates that appear in fewer than `candidate_pruning_min_df` sentences. Useful for filtering out very rare or noisy terms.
- **`max_df`**: Removes candidates that appear in more than `candidate_pruning_max_df` (as a percentage) of sentences. Useful for filtering out overly common, non-descriptive terms.

This is only applicable for `count` and `tfidf` tokenization.'''
        },
        validate=OneOf(['min_df', 'max_df', None]),
        allow_none=True,
    )
    candidate_pruning_min_df = fields.Integer(
        allow_none=True,
        metadata={
            'default': 1,
            'description': 'For `min_df` pruning, this is the minimum number of sentences a keyword candidate must appear in to be kept. A value of`1` means a candidate must appear in at least one sentence. A value of `2` helps filter out terms that appear in only a single sentence, which are often too specific or noisy.'
        }
    )
    candidate_pruning_max_df = fields.Float(
        allow_none=True,
        metadata={
            'default': 0.95,
            'description': 'For `max_df` pruning, this is the maximum proportion of sentences a keyword candidate can appear in. A value of `0.95` removes terms that are present in over 95% of sentences, as they are typically too common to be descriptive (e.g., domain-specific stop words).'
        }
    )
    rep_sentence_strategy = fields.String(
        metadata={
            'default': 'centroid',
            'description': '''Strategy for selecting the representative sentence for a cluster.

- **`centroid`**: Selects the sentence with the highest similarity to the cluster\'s geometric center (centroid). This sentence is the most "average" member.
- **`centrality`**: Selects the sentence with the highest average similarity to all *other* sentences in the cluster. This sentence is the most "well-connected" or central member, often a better summary if the cluster is not perfectly spherical.
'''
        },
        validate=OneOf(['centroid', 'centrality']),
    )
    label_fallback_strategy = fields.String(
        metadata={
            'default': 'rep_sentence_excerpt',
            'description': '''Defines the fallback behavior for labeling a cluster if no suitable keyword or predefined label is found.

- **`rep_sentence_excerpt`**: Uses an excerpt of the representative sentence as the label.
- **`highest_scoring_candidate`**: Forces the use of the top-ranked keyword candidate, even if its score is low.
- **`none`**: Leaves the cluster label empty. This is useful to clearly distinguish between successfully labeled and unlabeled clusters.
'''
        },
        validate=OneOf([
            'rep_sentence_excerpt', 'highest_scoring_candidate', 'none'
        ]),
    )
    include_candidate_details = fields.Boolean(
        metadata={
            'default': False,
            'description': 'If true, includes detailed statistics about the keyword candidate generation, pruning, and ranking process for each cluster. This is useful for debugging and fine-tuning keyword extraction.',
        },
    )


def _generate_plots_description():
    """
    Dynamically generates the description for the 'plots' field
    by introspecting the docstrings of the plot functions.
    """

    categories = {
        "General & Overview Plots": [
            "cluster-sizes", "embedding-histogram", "sentence-similarity-distribution",
            "wordcloud-combined", "wordcloud-overview", "wordcloud"
        ],
        "Cluster Structure & Cohesion Plots": [
            "cluster-scatter-pca", "cluster-scatter-tsne", "cluster-density", "silhouette",
            "cluster-cohesion-plot", "cluster-outlier-scores", "cluster-distance-distribution",
            "treemap", "dendrogram-heatmap"
        ],
        "Inter-Cluster Relationship Plots": [
            "cluster-separation", "cluster-overlap-graph", "sentence-bridge-identifier",
            "bipartite-cluster-sentence-graph"
        ],
        "Keyword & Topic Analysis Plots": [
            "keyword-contribution", "characteristic-keywords", "keyword-rarity-plot",
            "keyword-cooccurrence-graph", "keyword-cooccurrence-heatmap",
            "word-cooccurrence-graph", "word-cooccurrence-heatmap",
            "cluster-keyword-bipartite-graph", "keyword-trajectory",
            "label-word-semantic-hierarchy", "cluster-label-similarity-heatmap"
        ],
        "Advanced & Experimental Visualizations": [
            "topic-world-map", "galaxy-map-constellations", "semantic-gravity-well",
            "gravitational-force-network", "cluster-fingerprint-radar",
            "corpus-topic-radar", "cluster-aura-plot"
        ]
    }

    description_parts = ["A list of visualizations to generate for the clustering results. Each plot provides a different perspective on the data.\n"]

    for category, plot_names in categories.items():
        description_parts.append(f"### {category}")
        for plot_name in sorted(plot_names):
            if plot_name in _plotter_fn:
                func_doc = _plotter_fn[plot_name].__doc__
                if func_doc:
                    description_parts.append(f"- **`{plot_name}`**: {func_doc.strip().splitlines()[0]}")
    return '\n'.join(description_parts)


class SentenceClusterRequestOptions(Schema):
    algorithm = fields.String(
        metadata={
            'default': 'agglomerative',
            'description': '''The clustering algorithm to use.

- **`agglomerative`**: A hierarchical method that progressively merges clusters. It's a great general-purpose choice, especially when you have a target number of clusters (`n_clusters`) or a clear similarity cutoff (`distance_threshold`).

- **`optics`**: A density-based algorithm that excels at identifying clusters of varying densities and can separate noise points (outliers) that don't belong to any cluster. Use `optics` when:
    - You suspect your data contains irrelevant sentences that shouldn't be clustered.
    - You don't know the number of clusters beforehand.
    - Your clusters might be of different sizes and shapes.

Start with `agglomerative` for straightforward topic modeling. Switch to `optics` for noisy datasets or when outlier detection is important.
''',
        },
        validate=OneOf(['agglomerative', 'optics']),
    )
    n_clusters = fields.Integer(
        metadata={
            'description': '''The target number of clusters to form.

Use this when you have a specific number of topics you want to discover. This is the most direct way to control the output of `agglomerative` clustering. If set, `distance_threshold` is ignored. If you don't know the number of clusters, consider using `distance_threshold` or `n_clusters_search` instead.''',
            'default': None,
        },
        allow_none=True,
    )
    n_clusters_max = fields.Integer(
        metadata={
            'description': '''An optional upper bound for the number of clusters.

This is a safety mechanism to prevent an excessive number of clusters when using `distance_threshold`. It can be useful for very large or diverse datasets where a low threshold might otherwise create hundreds of small clusters. If set, it overrides `n_clusters` and `distance_threshold` if they would result in more clusters.''',
            'default': None,
        },
        allow_none=True,
    )
    n_clusters_search = fields.List(
        fields.Integer(),
        metadata={
            'description': '''A range `[min, max]` to search for the optimal number of clusters.

Use this when you want the API to automatically determine the best number of clusters within a given range. The system will test each number of clusters (`k`) from `min` to `max` and select the `k` that results in the highest `silhouette_score` (a measure of cluster quality). This is computationally more expensive but can produce better-defined clusters. This option overrides `n_clusters` and `distance_threshold`.''',
            'default': None,
        },
        allow_none=True,
    )
    distance_threshold = fields.Float(
        metadata={
            'default': 0.6,
            'description': '''A distance threshold that controls cluster granularity. The distance is calculated as `1 - cosine_similarity`, so a smaller value means higher similarity is required.

- For **`agglomerative`** clustering, this is the linkage distance threshold above which clusters will not be merged. It is only used when `n_clusters` is not set.
- For **`optics`** clustering, this value is used as `max_eps`, which defines the maximum distance between two samples for one to be considered as in the neighborhood of the other.

The effect is conceptually similar for both algorithms: a lower threshold creates more, smaller, and tighter clusters, while a higher threshold creates fewer, larger, and broader ones.

- **Low Threshold (e.g., `0.3` - `0.5`)**: Use this when you want to find very specific, tightly-related groups of sentences. This will result in more, smaller, and highly cohesive clusters. It's ideal for identifying distinct, narrow topics.
- **Medium Threshold (e.g., `0.6` - `0.7`)**: This is a good starting point for general-purpose topic modeling. It balances finding distinct topics without creating too many tiny, fragmented clusters. The default of `0.6` often provides a reasonable number of meaningful clusters.
- **High Threshold (e.g., `0.8` - `1.0`)**: Use this when you want to group sentences into a few broad, high-level categories. This will result in fewer, larger, and more diverse clusters. It's useful for getting a "big picture" overview of the main themes.

Start with the default of `0.6` and adjust based on the results. If you get too many small clusters, increase the threshold. If your clusters seem too broad and contain unrelated sentences, decrease it.
''',
            'example': 0.6,
        },
        allow_none=True,
    )
    linkage = fields.String(
        metadata={
            'default': 'average',
            'description': '''Which linkage criterion to use for `agglomerative` clustering. The linkage criterion determines how the distance between clusters is measured when deciding which clusters to merge.

- **`average`**: Uses the average of the distances between all pairs of sentences in the two clusters. Robust to noise and tends to produce balanced, cohesive clusters.

- **`complete`**: Uses the maximum distance between any two sentences in the two clusters. Tends to produce more compact, spherical clusters.

- **`single`**: Uses the minimum distance between any two sentences in the two clusters. Can handle non-elliptical shapes but is sensitive to noise and can lead to "chaining."

- **`ward`**: Minimizes the variance of the clusters being merged. Often produces well-separated, equally sized clusters. **Note**: `ward` linkage is only compatible with the `euclidean` metric.

Use `average` as a default. If you need very compact clusters, try `complete`. If you have well-defined, similarly-sized groups, `ward` (with `euclidean` metric) is a strong choice. Use `single` with caution.
''',
        },
        validate=OneOf(['ward', 'complete', 'average', 'single']),
    )
    metric = fields.String(
        metadata={
            'default': 'cosine',
            'description': '''The metric used to compute the distance between sentence embeddings.

- **`cosine`**: Measures the angle between two vectors, ignoring their magnitude. It is excellent for text embeddings because it captures semantic similarity regardless of sentence length. Distance is calculated as `1 - cosine_similarity`.

- **`euclidean`**: Measures the straight-line distance between two points (the tips of the vectors). It considers both direction and magnitude. For normalized embeddings (which this API uses), `euclidean` distance can yield similar results to `cosine`. It is required if using `ward` linkage.

- **`l1` / `manhattan`**: Measures distance by summing the absolute differences of the vector components.
- **`l2`**: Equivalent to `euclidean`.

Stick with `cosine` for most text clustering tasks. You must switch to `euclidean` if you choose to use `ward` linkage.
''',
        },
        validate=OneOf(['cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'precomputed']),
    )
    lang = fields.String(
        metadata={
            'default': None,
            'description': 'Language of the input text (e.g., `en`, `de`). If not provided, language detection is performed automatically on a sample of the input. Specifying the language improves performance by skipping this step.',
        },
        allow_none=True,
    )
    lang_detect_possible_locales = fields.List(
        fields.String(),
        metadata={
            'example': ['en', 'de', 'fr', 'es'],
            'description': 'When performing automatic language detection, this list limits the search to a specific set of languages. This can improve detection accuracy if you know the possible languages in your text. If empty, all supported languages are considered.',
        },
    )
    lang_detect_clean_text = fields.Boolean(metadata={
        'default': True,
        'description': 'Whether to clean the text (e.g., remove URLs, special characters) before language identification. Recommended for improving accuracy on noisy text.'
    })
    tokenize_pretokenized = fields.Boolean(metadata={
        'default': False,
        'description': 'Indicates if the input document is pre-tokenized. If true, sentences are assumed to be separated by two newlines (`\\n\\n`), which is faster than running the full sentence segmentation model.',
    })
    keywords = fields.Nested(SentenceClusterRequestKeywordsOptions())
    plots = fields.List(
        fields.String(
            validate=[
                OneOf(_plot_options),
            ],
        ),
        metadata={
            'uniqueItems': True,
            'example': [
                'cluster-sizes',
                'treemap',
                'cluster-overlap-graph',
                'galaxy-map-constellations',
                'bipartite-cluster-sentence-graph',
                # 'wordcloud', # Can be slow for many clusters
            ],
            'description': _generate_plots_description()
        },
        allow_none=True,
    )
    include_document = fields.Boolean(
        metadata={
            'default': False,
            'example': True,
            'description': 'If true, includes detailed document-level statistics in the response outcome, such as the overall silhouette score and cluster size distribution.'
        },
    )
    include_noise_cluster = fields.Boolean(
        metadata={
            'default': False,
            'description': 'If true, the noise cluster (cluster_id = -1) generated by OPTICS will be included in the response. By default, it is excluded.'
        },
    )

    @validates('plots')
    def no_duplicate_plots(self, value):
        if value is not None and len(value) != len(set(value)):
            raise ValidationError('plots must not contain duplicate items')


class SentenceClusterRequest(Schema):
    input = StringOrList(
        required=True,
        metadata={
            'description': 'Input text to be clustered. Can be a single string (document) or a list of strings (sentences). If a single string is provided, it will be split into sentences using Stanza.',
            'examples': [
                'Last summer, I spent three weeks in Italy exploring cities like Rome, Florence, and Venice. The trip was unforgettable, not only because of the historical sites but also because of the people I met along the way. In Rome, I interviewed a local tour guide named Alessandro Rossi, who had been working at the Colosseum for over fifteen years.\n\nAlessandro told me: \"Every day I see thousands of visitors from around the world. For me, the Colosseum is not just a monument; it’s a reminder of human creativity and resilience.\" His words made me reflect on how much history shapes our sense of identity.\n\nIn Florence, I attended an art workshop hosted by the Accademia di Belle Arti. There I met Professor Maria Bianchi, who specializes in Renaissance painting. She shared her perspective: \"Art is a dialogue between centuries. When students today look at works by Michelangelo or Botticelli, they are not just seeing old paintings — they are entering a conversation that started hundreds of years ago.\" I found her passion inspiring.\n\nFood was another highlight of my journey. At a small trattoria near the Ponte Vecchio, I tasted handmade gnocchi paired with Chianti wine. The owner, Signora Lucia, told me proudly that her recipes had been passed down through her family since the 1920s. \"Cooking is memory,\" she said. \"Every dish carries a story.\" I couldn’t agree more.\n\nWhen I moved on to Venice, I had the chance to interview Anna and Marco, a young couple running a gondola business. They explained that tourism had been deeply affected by the pandemic in 2020, but they were slowly rebuilding. \"Venice will always be Venice,\" Marco said with a smile, guiding his gondola through the Grand Canal. \"Even if times are tough, the city finds a way to survive.\"\n\nOutside of Italy, I also traveled earlier this year to New York City for a conference on digital media hosted by Columbia University. During the event in March 2023, I sat down with journalist David Chen from The New York Times. I asked him about the future of independent publishing, and he replied: \"Blogs and newsletters have given writers a new sense of freedom. Readers today want authenticity, and personal voices matter more than ever.\"\n\nAfter the conference, I visited the Metropolitan Museum of Art, where a special exhibit on Japanese calligraphy was on display. I spent nearly two hours walking through the galleries, marveling at works borrowed from Kyoto and Tokyo. A curator named Keiko Yamamoto explained to me that the exhibit had been organized in partnership with the Tokyo National Museum. \"We want to build cultural bridges,\" she said, \"so that traditions do not remain isolated but are shared globally.\"\n\nOn a more personal note, I’ve recently been working on balancing my professional life as a freelance writer with personal health. I spoke with Dr. Susan Patel, a nutritionist in San Francisco, who encouraged me to adopt a Mediterranean-style diet. \"It’s not just about living longer,\" she emphasized, \"but about living better.\" Since following her advice, I’ve felt more energized during my daily jogs at Golden Gate Park.\n\nLooking forward, I plan to attend the Frankfurt Book Fair in October 2024. I’m excited to reconnect with authors I met in Berlin last year, including novelist Clara Weiss, whose book on post-war Germany was shortlisted for the Deutscher Buchpreis. Clara told me during our last conversation: \"Literature is a mirror, but also a bridge — it reflects our struggles and connects us to others.\" Her words still resonate with me today.\n\nTo close this blog entry, I’d like to share one last thought from an interview I conducted with musician Alejandro García in Barcelona. When I asked him about the role of music in society, he smiled and strummed his guitar: \"Music is the heartbeat of culture. Without it, we forget how to feel.\" That sentence has stayed with me, and I think about it every time I press play on my Spotify playlist while working on new articles.\n\n',
                'Reading novels is one of my favorite ways to relax after a busy day.\n\nWhenever I finish work, I like to unwind with a good book, especially fiction.\n\nScience fiction stories inspire me to imagine futuristic worlds and technologies.\n\nI find that sci-fi books expand my creativity and curiosity about the universe.\n\nHistory books fascinate me because they offer insights into how societies developed.\n\nLearning about the past through biographies and historical accounts helps me understand the present.\n\nI sometimes enjoy poetry, though I often need to read a poem several times to truly appreciate it.\n\nPoetry allows me to connect with emotions and ideas in a more abstract, artistic way.\n\nCooking is another activity that brings me joy, especially when trying new recipes.\n\nExperimenting in the kitchen often feels like a creative science experiment with delicious results.\n\nTraveling to different countries introduces me to unique flavors and traditional dishes.\n\nExploring local food markets during travel is one of the most memorable experiences for me.\n\n',
                [
                    "Early computing relied heavily on rigid data structures such as fixed-length records. These structures offered predictability but were inflexible when dealing with evolving requirements.",
                    "The rise of self-describing formats like JSON and XML introduced more flexibility. Instead of strict layouts, data could be exchanged with embedded metadata about its shape and meaning.",
                    "JSON Schema grew out of this need for shared agreement. It provided a language to describe what a valid JSON document should look like, allowing systems to validate inputs consistently.",
                    "Meanwhile, in databases, schemas had long existed as formal contracts. Relational systems enforced strict typing and constraints, while newer document stores often relaxed schema enforcement for agility.",
                    "Serialization formats like Protocol Buffers and Avro took a different path. They prioritized compactness and speed, defining schemas in separate interface files rather than embedding them directly in the data.",
                    "Schema evolution is a recurring challenge. How do you safely change a field name, adjust a type, or deprecate a property without breaking existing consumers of the data?",
                    "Some ecosystems embrace schema-less approaches, relying instead on application logic to handle unexpected data. This trades validation for flexibility but often creates hidden complexity.",
                    "GraphQL introduced a new way of thinking. Rather than describing the shape of documents, it defined a query language with strong typing, allowing clients to request exactly what they need.",
                    "In the world of APIs, schemas serve as contracts between teams. They reduce ambiguity by specifying not just data types but also constraints, defaults, and expected patterns.",
                    "Tooling has become a major factor in adoption. JSON Schema validators, linters, and code generators help automate repetitive tasks and catch errors earlier in development.",
                    "From a theoretical angle, schemas resemble grammars in formal language theory. Both define valid structures in a symbolic system, whether those symbols are words or key-value pairs.",
                    "Practical schema use often mixes validation with documentation. Developers annotate fields with descriptions, examples, and usage hints, blending technical rules with human-readable guidance.",
                    "A recurring debate is whether strict schemas stifle innovation. Supporters argue they provide guardrails for quality, while critics prefer looser models that allow rapid iteration.",
                    "In distributed systems, schemas play an even greater role. Without a shared agreement, microservices risk sending incompatible data and breaking integrations.",
                    "One important machine-automatable use of schemas is dynamic form generation. Given a schema that describes fields, types, and constraints, a system can render a user interface automatically without manual coding.",
                    "Form generation extends beyond layout. A schema can define select options, input ranges, default values, and validation rules, ensuring the UI and the backend share the same contract.",
                    "This ability also powers configuration tools. By reading a schema, applications can build editors for settings or workflows, allowing non-developers to interact safely with structured data.",
                    "Automation goes further when schemas integrate with testing. Test suites can auto-generate input cases by walking through a schema and checking boundary conditions for each property.",
                    "Documentation platforms benefit too. Many API portals rely on schemas to auto-generate human-readable reference material, turning machine definitions into guides for developers.",
                    "Schema-driven design is particularly strong in low-code platforms. By leveraging schemas, these systems let users assemble workflows and UIs from components without writing code directly.",
                    "The combination of schemas and UI generation also improves accessibility. Since form definitions are explicit, assistive technologies can interpret field roles, labels, and constraints with greater accuracy.",
                    "Schemas even influence data visualization. When a dataset has a clear description of types and formats, tools can propose charts or tables automatically, reducing setup overhead.",
                    "Looking ahead, hybrid models may emerge. These could combine strong schema guarantees where stability is crucial with looser contracts in areas where experimentation is valuable.",
                ],
            ],
        }
    )
    options = fields.Nested(SentenceClusterRequestOptions())
    model = fields.String(
        validate=OneOf([model.id for model in models.list() if issubclass(model, ModelBase) and 'vector' in model.tasks and model.modality and 'text' in model.modality]),
        metadata={
            'default': 'text',
            'description': 'The ID of the sentence transformer model to use for generating embeddings. This model converts text into numerical vectors, which are then used for clustering. You can list available models via the `/models` endpoint.',
        },
        allow_none=True,
    )


class ClusterKeyword(Schema):
    phrase = fields.String()
    sim = fields.Float()
    freq_or_tfidf = fields.Float(allow_none=True)
    score = fields.Float()


class ClusterCandidateDetails(Schema):
    initial_candidate_count = fields.Integer(description="Total number of unique candidates considered for this cluster before any filtering.")
    final_candidate_count = fields.Integer(description="Number of candidates remaining after filtering and selection for scoring.")
    label_candidate_count = fields.Integer(description="Number of candidates considered for the main cluster label.")
    keyword_candidate_count = fields.Integer(description="Number of candidates considered for the keyword list.")
    candidates = fields.List(
        fields.Dict(
            keys=fields.String(),
            values=fields.Raw(),
            description="A dictionary containing detailed scoring information for a single candidate."
        ),
        description="A list of all candidates that were scored, including their individual scores (semantic, frequency, lift, etc.), ranks, and whether they were sourced locally, globally, or from a predefined list."
    )


class ClusterOutcome(Schema):
    cluster_id = fields.Integer()
    size = fields.Integer()
    label = fields.String()
    label_score = fields.Float(allow_none=True)
    label_type = fields.String()
    keywords = fields.List(fields.Nested(ClusterKeyword()))
    rep_id = fields.Integer()
    entities = fields.List(fields.Dict())
    lang = fields.String()
    sentence_ids = fields.List(fields.Integer())
    sentences = fields.List(fields.String())
    sentences_centroid_similarity = fields.List(fields.Float(allow_none=True), allow_none=True)
    radius = fields.Float()
    avg_dist = fields.Float()
    diameter = fields.Float()
    candidate_details = fields.Nested(ClusterCandidateDetails(), allow_none=True)


class DocumentStats(Schema):
    token_count = fields.Integer(description="Total number of tokens in the input document.")
    sentence_count = fields.Integer(description="Total number of sentences clustered.")
    cluster_count = fields.Integer(description="Total number of clusters identified.")
    noise_points = fields.Integer(description="Number of sentences not assigned to any cluster (noise). Only applicable for density-based algorithms like OPTICS.")
    silhouette_score = fields.Float(allow_none=True, description="Overall silhouette score for the clustering, indicating cluster separation. Higher is better. Requires at least 2 clusters.")
    avg_cluster_size = fields.Float(description="Average number of sentences per cluster.")
    cluster_size_std_dev = fields.Float(description="Standard deviation of cluster sizes, indicating how balanced the clusters are.")


class SentenceClusterOutcome(Schema):
    clusters = fields.List(fields.Nested(ClusterOutcome()))
    tree = fields.Dict()
    entities = fields.Dict()
    document = fields.Nested(DocumentStats(), allow_none=True)


class SentenceClusterResponse(InferBaseResponse):
    outcome = fields.Nested(SentenceClusterOutcome())
    assets = fields.List(fields.Dict())


def api_clustering(app: APIFlask, s: Services):
    @app.route(f'/sentence-clusters', methods=['POST'])
    @app.input(SentenceClusterRequest)
    @app.output(SentenceClusterResponse())
    @app.doc(tags=[f'NLP'], description='''
This API endpoint performs sentence clustering using various techniques, including semantic similarity and keyword extraction.

It groups similar sentences together and provides insights into each cluster, such as representative keywords, entities, and cohesion metrics.

The process involves:

1. **Sentence Segmentation**: If a document is provided, it's split into individual sentences using Stanza.
2. **Embedding Generation**: Sentences are converted into vector embeddings using a pre-trained model (e.g., Sentence Transformers).
3. **Agglomerative Clustering**: Sentences are clustered based on the similarity of their embeddings.
4. **Keyword Extraction**: For each cluster, relevant keywords are extracted using either CountVectorizer, TF-IDF, or Stanza's NLP capabilities (for noun phrases and named entities).
5. **Cluster Labeling**: Clusters are assigned a label based on either the most representative keyword or a predefined label if provided and a sufficient similarity threshold is met.
6. **Cohesion Metrics**: Calculates metrics like radius, average distance, and diameter to describe the compactness and spread of clusters.
7. **Visualizations**: Generates various plots and charts (e.g., scatter plots, heatmaps, word clouds) to help visualize the clusters and their characteristics.

**Input**: A single document string or a list of sentence strings.

**Output**: A list of identified clusters, each with its size, label, keywords, representative sentence, and various metrics.

Also returns a list of assets which are base64 encoded images for visualization.

> Note that keyword options do not influence how cluster are formed, only how they are labelled, their keywords and plots like word clouds.

**TODO**:
- Refine and add more detailed API specs; this is currently more of an experiment.
- Usage, tokens, and durations are not correctly collected. Full tracking needs to be added when this is refined, possibly with a new aggregation for multi-stage steps (e.g., per-sentence processing).
''')
    def sentence_clusters(json_data):
        input = json_data['input']
        options = json_data.get('options', {}) or {}
        if not isinstance(input, List) and not isinstance(input, str):
            raise HTTPError(400, 'input must be array of sentences or a document')

        lang = options.get('lang')

        infer_res = InferTracker()
        model_id = json_data.get('model', None) or 'text'
        model, tracker = models.get_tracked(model_id, infer_res)

        if not lang:
            on_loaded_lang_detect = tracker('load_stanza_lang_detect')
            pipe_lang_detect = stanza_model.pipeline(
                locale="multilingual",
                langid_lang_subset=options.get('lang_detect_possible_locales'),
                langid_clean_text=options.get('lang_detect_clean_text', True),
            )
            on_loaded_lang_detect()
            on_lang_detect = tracker('stanza_lang_detect')
            sample = input[0] if isinstance(input, List) else input
            doc: Document = pipe_lang_detect(sample, 'langid')
            lang = doc.lang
            on_lang_detect()

        keywords_options = options.get('keywords', {}) or {}
        tokenization_choice = keywords_options.get('tokenization', 'tfidf')

        corpus: List[str] = []
        stanza_docs: List[Document] = []
        if isinstance(input, str):
            on_stanza_loaded = tracker('load_stanza_preprocess')
            pipe = stanza_model.pipeline(
                locale=lang,
                tokenize_no_ssplit=False,  # always split sentences for document input
                tokenize_pretokenized=options.get('tokenize_pretokenized', False),
                processors='tokenize',
            )
            on_stanza_loaded()

            on_processed_stanza = tracker('stanza_preprocess')
            doc: Document = pipe(input)
            on_processed_stanza(tokens=doc.num_tokens)

            sentences: List[Sentence] = doc.sentences
            for sent in sentences:
                corpus.append(sent.text)
                stanza_doc = Document([], text=sent.text)
                stanza_doc.sentences.append(sent)
                stanza_docs.append(stanza_doc)
        else:
            corpus = input
            on_stanza_loaded = tracker('load_stanza_preprocess')
            pipe = stanza_model.pipeline(locale=lang, processors='tokenize', tokenize_no_ssplit=True)
            on_stanza_loaded()

            on_processed_stanza = tracker('stanza_preprocess')
            # Use bulk_process for efficiency on list of sentences
            stanza_docs = pipe.bulk_process(corpus)
            on_processed_stanza(tokens=sum(doc.num_tokens for doc in stanza_docs))

        if len(corpus) < 2:
            raise HTTPError(400, message='Input must contain more than one sentence for clustering.')

        on_encode_corpus = tracker('encode_corpus')
        used_tokens, corpus_embeddings = model.encode_with_stats(corpus, batch_size=96)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        on_encode_corpus(tokens=used_tokens)

        n_clusters: Union[int, None] = options.get('n_clusters', None)
        n_clusters_max: Union[int, None] = options.get('n_clusters_max', None)
        n_clusters_search: Union[List[int], None] = options.get('n_clusters_search', None)
        distance_threshold = options.get('distance_threshold', 0.6)
        linkage = options.get('linkage', 'average')
        metric = options.get('metric', 'cosine')

        if n_clusters_max is not None:
            n_clusters = min(n_clusters, n_clusters_max) if n_clusters is not None else min(n_clusters_max, len(corpus))

        if n_clusters_search and len(n_clusters_search) == 2:
            on_search_clusters = tracker('search_optimal_clusters')
            min_k, max_k = n_clusters_search
            max_k = min(max_k, len(corpus) - 1)
            best_score = -1
            best_k = min_k

            for k in range(min_k, max_k + 1):
                if k < 2: continue
                temp_model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)
                labels = temp_model.fit_predict(corpus_embeddings)
                score = silhouette_score(corpus_embeddings, labels, metric=metric)
                if score > best_score:
                    best_score = score
                    best_k = k
            n_clusters = best_k
            distance_threshold = None
            on_search_clusters()
        elif n_clusters is None and distance_threshold is None:
            distance_threshold = 0.6
        elif n_clusters is not None and distance_threshold is not None:
            distance_threshold = None
        elif n_clusters is not None:
            if n_clusters < 2:
                raise HTTPError(400, message='n_clusters must be at least 2 for clustering.')
            if n_clusters == len(corpus):
                cluster_assignment = np.arange(len(corpus))
            if n_clusters > len(corpus):
                raise HTTPError(400, message='n_clusters must be less than the number of sentences.')
            distance_threshold = None

        if n_clusters is None and distance_threshold is None:
            raise HTTPError(400, message='Either n_clusters or distance_threshold must be specified for clustering.')

        on_computed_cluster = tracker('compute_cluster')

        algorithm = options.get('algorithm', 'agglomerative')
        if algorithm == 'optics':
            clustering_model = OPTICS(
                min_samples=max(2, int(len(corpus) * 0.05)),  # 5% of corpus size as min points
                metric=metric,
                max_eps=distance_threshold,
                cluster_method='dbscan'
            )
        else:
            clustering_model = AgglomerativeClustering(
                n_clusters=n_clusters, distance_threshold=distance_threshold, linkage=linkage, metric=metric
            )

        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        # Post-hoc check for n_clusters_max when distance_threshold is used (should not be needed, as done before)
        # if n_clusters_max is not None and n_clusters is None and distance_threshold is not None:
        #     num_found_clusters = len(set(cluster_assignment))
        #     if num_found_clusters > n_clusters_max:
        #         clustering_model = AgglomerativeClustering(n_clusters=n_clusters_max, linkage=linkage, metric=metric)
        #         clustering_model.fit(corpus_embeddings)
        #         cluster_assignment = clustering_model.labels_

        on_computed_cluster()

        clusters = {}
        for idx, c_id in enumerate(cluster_assignment):
            clusters.setdefault(c_id, {
                'indices': [],
                'sentences': [],
                'embeddings': [],
            })
            clusters[c_id]['indices'].append(idx)
            clusters[c_id]['sentences'].append(corpus[idx])
            clusters[c_id]['embeddings'].append(corpus_embeddings[idx])

        labeled_clusters, candidate_meta_by_cluster = _process_clusters(tracker, model, options, clusters, lang, stanza_docs)

        plots: Union[List[str], None] = options.get('plots')

        vectorizer = None
        tfidf_matrix = None
        keywords_options = options.get('keywords', {})
        plots_requiring_tfidf = {'keyword-contribution', 'characteristic-keywords'}
        if plots and any(p in plots_requiring_tfidf for p in plots):
            stop_words_list = []
            if keywords_options.get('stop_words') == 'english':
                stop_words_list = list(ENGLISH_STOP_WORDS)
            elif keywords_options.get('stop_words'):
                stop_words_list = keywords_options.get('stop_words')

            vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words=stop_words_list)
            tfidf_matrix = vectorizer.fit_transform(corpus)

        assets: Union[List, None] = None
        if plots and labeled_clusters:
            assets = make_cluster_plots(
                plots=plots,
                corpus_embeddings=corpus_embeddings,
                cluster_assignment=cluster_assignment,
                labeled_clusters=labeled_clusters,
                clusters=clusters,
                tracker=tracker,
                model=model,
                clustering_model=clustering_model,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                candidate_meta_by_cluster=candidate_meta_by_cluster,
                corpus=corpus,
                stanza_docs=stanza_docs,
                options=options,
            )

        document_stats = None
        if options.get('include_document', False):
            num_clusters = len(set(c for c in cluster_assignment if c != -1))
            cluster_sizes = [lc['size'] for lc in labeled_clusters if lc['cluster_id'] != -1]
            overall_silhouette_score = None
            if num_clusters >= 2:
                try:
                    overall_silhouette_score = silhouette_score(corpus_embeddings, cluster_assignment, metric=metric)
                except ValueError:
                    overall_silhouette_score = None  # Not enough samples in a cluster

            total_tokens = sum(
                meta.get('token_count', 0) for meta in candidate_meta_by_cluster.values() if meta and isinstance(meta, dict)
            )

            document_stats = {
                'token_count': total_tokens,
                'sentence_count': len(corpus),
                'cluster_count': num_clusters,
                'noise_points': np.sum(cluster_assignment == -1),
                'silhouette_score': float(overall_silhouette_score) if overall_silhouette_score is not None else None,
                'avg_cluster_size': float(np.mean(cluster_sizes)) if cluster_sizes else 0,
                'cluster_size_std_dev': float(np.std(cluster_sizes)) if cluster_sizes else 0,
            }

        # Filter out noise cluster (-1) from the main list if it exists
        if algorithm == 'optics' and not options.get('include_noise_cluster', False):
            labeled_clusters = [lc for lc in labeled_clusters if lc['cluster_id'] != -1]

        tree = {}
        entities = {}
        for lc in labeled_clusters:
            label_key = lc['label'] if lc['label'] else 'Unlabeled'
            if label_key not in tree:
                tree[label_key] = {
                    'count_sent': 0,
                    'clusters': [],
                    'keywords': [],
                    'entities': [],
                }
            tree[label_key]['count_sent'] += lc['size']
            tree[label_key]['clusters'].append({
                'cluster_id': lc['cluster_id'],
                'size': lc['size'],
                'rep_id': lc['rep_id'],
                'sentence_ids': lc['sentence_ids'],
            })
            for kw in lc['keywords']:
                if kw['phrase'] not in tree[label_key]['keywords']:
                    tree[label_key]['keywords'].append(kw['phrase'])
            if lc['entities']:
                for ent in lc['entities']:
                    if ent['text'] not in tree[label_key]['entities']:
                        tree[label_key]['entities'].append(ent['text'])
                        if ent['text'] not in entities:
                            entities[ent['text']] = {'cluster_ids': set(), 'sentence_ids': set()}
                        entities[ent['text']]['cluster_ids'].add(lc['cluster_id'])
                        entities[ent['text']]['sentence_ids'].update(lc['sentence_ids'])

        for ent_text, ent_data in entities.items():
            entities[ent_text]['cluster_ids'] = sorted(list(ent_data['cluster_ids']))
            entities[ent_text]['sentence_ids'] = sorted(list(ent_data['sentence_ids']))

        return {
            'usage': infer_res.usage,
            'outcome': {
                'tree': tree,
                'entities': entities,
                'document': document_stats,
                'clusters': labeled_clusters,
            },
            'assets': assets
        }


def _process_clusters(
    tracker,
    model,
    options,
    clusters, lang,
    all_stanza_docs: List[Document]
):
    keywords_options = options.get('keywords', {}) or {}
    candidate_sources = keywords_options.get('candidate_sources', ['global', 'local'])

    tokenization_choice = keywords_options.get('tokenization', 'tfidf')
    ngram_range = tuple(keywords_options.get('ngram_range', [1, 3]))
    stop_words = []
    if keywords_options.get('stop_words') == 'english':
        stop_words = list(ENGLISH_STOP_WORDS)
    elif keywords_options.get('stop_words'):
        stop_words = keywords_options.get('stop_words')
    else:
        stop_words = [
            # a very reduced set of the most noisy stop word
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for',
            'if', 'in', 'into', 'is', 'it', 'no', 'not',
            'has', 'had', 'have', 'having',
            'of', 'on', 'or', 'such', 'that', 'the', 'their', 'them', 'then', 'than', 'there', 'these', 'they', 'this', 'those', 'to', 'too',
            'so', 'us', 'very', 'was', 'we', 'well', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
            'he', 'she', 'you', 'his', 'her', 'him', 'himself', 'herself', 'its', 'itself', 'our', 'ours', 'ourselves', 'your', 'yours', 'yourself', 'yourselves',
            'from', 'here', 'how', 'just', 'may', 'must', 'out', 'said', 'say', 'says', 'should',
        ]

    max_candidates = keywords_options.get('max_candidates')
    candidate_selection_strategy = keywords_options.get('candidate_selection_strategy', 'proportional')
    proportional_ratio = float(keywords_options.get('proportional_ratio', 0.6))
    proportional_min = keywords_options.get('proportional_min', None)
    proportional_max = int(keywords_options.get('proportional_max', 1000))
    proportional_max_local = keywords_options.get('proportional_max_local')
    proportional_max_global = keywords_options.get('proportional_max_global')
    density_base = int(keywords_options.get('density_base', 100))
    density_scale = int(keywords_options.get('density_scale', 400))
    log_scale_factor = int(keywords_options.get('log_scale_factor', 100))
    log_base = float(keywords_options.get('log_base', 10.0))
    density_max_dist = float(keywords_options.get('density_max_dist', 0.5))
    tfidf_adaptive_percentile = int(keywords_options.get('tfidf_adaptive_percentile', 80))

    top_keywords = int(keywords_options.get('top_keywords', 3))
    semantic_weight = float(keywords_options.get('semantic_weight', 0.75))
    freq_weight = float(keywords_options.get('freq_weight', 0.25))
    basic_labels = keywords_options.get('labels')
    predefined_labels = keywords_options.get('predefined_labels')
    predefined_keywords = keywords_options.get('predefined_keywords')
    thresholds = keywords_options.get('thresholds', {}) or {}
    predefined_label_threshold = float(thresholds.get('predefined_label', 0.32))
    mmr_lambda = float(keywords_options.get('mmr_lambda', 0.0))  # 0 = no MMR, >0 provides simple diversity
    lift_weight = float(keywords_options.get('lift_weight', 0.0))
    global_penalty_factor = float(keywords_options.get('global_penalty_factor', 1.0))
    stanza_processors = keywords_options.get('stanza_processors', 'tokenize,pos')
    max_features = keywords_options.get('max_features')
    candidate_pruning_strategy = keywords_options.get('candidate_pruning_strategy')
    candidate_pruning_min_df = keywords_options.get('candidate_pruning_min_df', 2)
    candidate_pruning_max_df = keywords_options.get('candidate_pruning_max_df', 0.95)

    rep_sentence_strategy = keywords_options.get('rep_sentence_strategy', 'centroid')
    label_fallback_strategy = keywords_options.get('label_fallback_strategy', 'rep_sentence_excerpt')
    dynamic_candidate_source = keywords_options.get('dynamic_candidate_source', ['labels', 'keywords'])
    force_predefined_for = keywords_options.get('force_predefined_for', None)

    include_candidate_details = keywords_options.get('include_candidate_details', False)
    predefined_emb = None
    final_predefined_labels = []

    if predefined_labels:
        final_predefined_labels = list(predefined_labels.keys())
        if basic_labels:
            # Add any labels from `labels` that are not in `predefined_labels`
            for label in basic_labels:
                if label not in final_predefined_labels:
                    final_predefined_labels.append(label)
        try:
            on_encode_labels = tracker('encode_labels')

            all_texts_to_embed = []
            label_to_indices = {}
            current_idx = 0
            for label in final_predefined_labels:
                examples = [label] + predefined_labels.get(label, [])
                start_idx = current_idx
                end_idx = start_idx + len(examples)
                label_to_indices[label] = (start_idx, end_idx)
                all_texts_to_embed.extend(examples)
                current_idx = end_idx

            used_tokens, all_example_embs = model.encode_with_stats(all_texts_to_embed, batch_size=256)
            all_example_embs_np = np.array(all_example_embs)

            # Create concept vectors by averaging
            all_label_embeddings = [
                np.mean(all_example_embs_np[label_to_indices[label][0]:label_to_indices[label][1]], axis=0, keepdims=True)
                for label in final_predefined_labels
            ]

            predefined_emb = np.vstack(all_label_embeddings)
            predefined_emb = predefined_emb / np.linalg.norm(predefined_emb, axis=1, keepdims=True)
            on_encode_labels(tokens=used_tokens)
        except Exception:
            predefined_emb = None
    elif basic_labels:
        final_predefined_labels = basic_labels
        try:
            on_encode_labels = tracker('encode_labels')
            used_tokens, predefined_emb = model.encode_with_stats(basic_labels, batch_size=256)
            predefined_emb = predefined_emb / np.linalg.norm(predefined_emb, axis=1, keepdims=True)
            on_encode_labels(tokens=used_tokens)
        except Exception:
            predefined_emb = None

    stanza_nlp = None
    vectorizer = None
    lang_used = lang
    if tokenization_choice == 'stanza':
        if not lang_used and clusters:
            if all_stanza_docs:
                lang_used = all_stanza_docs[0].lang

        if lang_used:
            try:
                stanza_nlp = stanza_model.pipeline(locale=lang_used, processors=stanza_processors, tokenize_no_ssplit=True)
                on_stanza_adv_process = tracker('stanza_deep_process')
                stanza_nlp(all_stanza_docs)
                on_stanza_adv_process()
            except Exception:
                stanza_nlp = None
    elif tokenization_choice == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words=stop_words, max_features=max_features)
    else:
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words, max_features=max_features)
    global_candidates = []
    global_candidate_data = {}
    if 'global' in candidate_sources and (not dynamic_candidate_source or 'labels' in dynamic_candidate_source or 'keywords' in dynamic_candidate_source):
        on_build_global_candidates = tracker('build_global_candidates')
        all_sentences = [doc.text for doc in all_stanza_docs]
        global_candidate_data = _build_candidates_vectorizer(
            sentences=all_sentences,
            stanza_docs=all_stanza_docs,
            tokenization_choice=tokenization_choice,
            max_candidates=max_candidates,
            candidate_selection_strategy=candidate_selection_strategy,
            proportional_ratio=proportional_ratio,
            proportional_min=proportional_min,
            proportional_max=proportional_max_global or proportional_max,
            log_scale_factor=log_scale_factor,
            log_base=log_base,
            tfidf_adaptive_percentile=tfidf_adaptive_percentile,
            ngram_range=ngram_range,
            stop_words=stop_words,
            max_features=max_features,
            vectorizer=vectorizer,
            candidate_pruning_strategy=candidate_pruning_strategy,
            candidate_pruning_min_df=candidate_pruning_min_df,
            candidate_pruning_max_df=candidate_pruning_max_df,
            is_global=True,
            density_max_dist=density_max_dist,
            cluster_avg_dist=None,
            density_scale=density_scale,
            density_base=density_base,
        )
        global_candidates = global_candidate_data.get('candidates', [])
        on_build_global_candidates()

    # Generate local candidates for each cluster to ensure specificity.
    cluster_metrics = {}
    if candidate_selection_strategy == 'density':
        for cid, c_dict in clusters.items():
            embeddings = np.vstack(c_dict['embeddings'])
            centroid = embeddings.mean(axis=0, keepdims=True)
            sims = cosine_similarity(embeddings, centroid).ravel()
            avg_dist = float(1 - sims.mean())
            cluster_metrics[cid] = {'avg_dist': avg_dist}
    on_build_local_candidates = tracker('build_local_candidates')
    all_unique_candidates = set(global_candidates)
    cluster_candidate_data = {}
    for cluster_id, cluster_dict in clusters.items():
        if 'local' in candidate_sources and (not dynamic_candidate_source or 'labels' in dynamic_candidate_source or 'keywords' in dynamic_candidate_source):
            # Get the pre-processed stanza docs for this specific cluster
            cluster_stanza_docs = [all_stanza_docs[i] for i in cluster_dict['indices']]
            local_data = _build_candidates_vectorizer(
                sentences=cluster_dict['sentences'],
                stanza_docs=cluster_stanza_docs,
                tokenization_choice=tokenization_choice,
                max_candidates=max_candidates,
                candidate_selection_strategy=candidate_selection_strategy,
                proportional_ratio=proportional_ratio,
                proportional_min=proportional_min,
                proportional_max=proportional_max_local or proportional_max,
                density_base=density_base,
                density_scale=density_scale,
                log_scale_factor=log_scale_factor,
                log_base=log_base,
                density_max_dist=density_max_dist,
                cluster_avg_dist=cluster_metrics.get(cluster_id, {}).get('avg_dist'),
                tfidf_adaptive_percentile=tfidf_adaptive_percentile,
                ngram_range=ngram_range,
                stop_words=stop_words,
                max_features=max_features,
                vectorizer=vectorizer,
                candidate_pruning_strategy=candidate_pruning_strategy,
                candidate_pruning_min_df=candidate_pruning_min_df,
                candidate_pruning_max_df=candidate_pruning_max_df,
                is_global=False,
            )
            combined_candidates = set(global_candidates)
            combined_candidates.update(local_data.get('candidates', []))
            local_data['candidates'] = list(combined_candidates)
            cluster_candidate_data[cluster_id] = local_data
            all_unique_candidates.update(local_data['candidates'])
        else:
            cluster_candidate_data[cluster_id] = {
                'candidates': global_candidates,
                'all_phrases': global_candidates,
                'token_count': int(global_candidate_data.get('token_count', 0) * (len(cluster_dict['sentences']) / len(all_stanza_docs))) if all_stanza_docs else 0,
                'col_sums': global_candidate_data.get('col_sums'),
                'phrase_to_idx': global_candidate_data.get('phrase_to_idx'),
            }
    on_build_local_candidates()

    candidate_embeddings_map = {}
    if predefined_keywords:
        on_encode_predefined_keywords = tracker('encode_predefined_keywords')

        all_texts_to_embed = []
        keyword_to_indices = {}
        current_idx = 0
        for keyword, examples in predefined_keywords.items():
            all_unique_candidates.add(keyword)
            texts_to_embed = [keyword] + examples
            start_idx = current_idx
            end_idx = start_idx + len(texts_to_embed)
            keyword_to_indices[keyword] = (start_idx, end_idx)
            all_texts_to_embed.extend(texts_to_embed)
            current_idx = end_idx

        used_tokens, all_example_embs = model.encode_with_stats(all_texts_to_embed, batch_size=128)
        all_example_embs_np = np.array(all_example_embs)

        for keyword, (start, end) in keyword_to_indices.items():
            concept_vec = np.mean(all_example_embs_np[start:end], axis=0)
            candidate_embeddings_map[keyword] = concept_vec
        on_encode_predefined_keywords(tokens=used_tokens)

    if candidate_embeddings_map:
        # De-normalize for a moment to add new ones, then re-normalize all
        for key, vec in candidate_embeddings_map.items():
            norm = np.linalg.norm(vec)
            if norm > 0:
                candidate_embeddings_map[key] = vec / norm

    if all_unique_candidates:
        unique_candidates_list = list(all_unique_candidates)
        on_encode_candidates = tracker('encode_candidates')
        used_tokens, all_cand_emb = model.encode_with_stats(unique_candidates_list, batch_size=256)
        all_cand_emb_norm = all_cand_emb / np.linalg.norm(all_cand_emb, axis=1, keepdims=True)
        for i, candidate in enumerate(unique_candidates_list):
            candidate_embeddings_map[candidate] = all_cand_emb_norm[i]
        on_encode_candidates(tokens=used_tokens)

    on_process_clusters = tracker('process_clusters')
    labeled_clusters = []
    candidate_meta_by_cluster = {}

    for cluster_id, cluster_dict in clusters.items():
        cluster_stanza_docs = [all_stanza_docs[i] for i in cluster_dict['indices']] if tokenization_choice == 'stanza' else None
        meta, top_phrases, centroid, sims, candidate_meta, candidate_details = _top_phrases_for_cluster(
            np.vstack(cluster_dict['embeddings']),
            top_keywords=top_keywords,
            semantic_weight=semantic_weight,
            freq_weight=freq_weight,
            mmr_lambda=mmr_lambda,
            lift_weight=lift_weight,
            global_penalty_factor=global_penalty_factor,
            global_candidate_data=global_candidate_data,
            precomputed_candidates_data=cluster_candidate_data.get(cluster_id, {}),
            rep_sentence_strategy=rep_sentence_strategy,
            candidate_embeddings_map=candidate_embeddings_map,
            predefined_keywords=predefined_keywords,
            dynamic_candidate_source=dynamic_candidate_source,
            stanza_docs=cluster_stanza_docs,
            force_predefined_for=force_predefined_for,
            thresholds=thresholds,
            include_candidate_details=include_candidate_details,
        )
        candidate_meta_by_cluster[cluster_id] = candidate_meta

        chosen_label = None
        label_score = None
        label_type = 'candidate'
        if final_predefined_labels and predefined_emb is not None and centroid is not None:
            pre_sims = cosine_similarity(predefined_emb, centroid).ravel()
            best_pre_idx = int(np.argmax(pre_sims))
            best_pre_sim = float(pre_sims[best_pre_idx])
            if force_predefined_for and 'labels' in force_predefined_for or best_pre_sim >= predefined_label_threshold:
                chosen_label = final_predefined_labels[best_pre_idx]
                label_score = best_pre_sim
                label_type = 'predefined'

        rep_idx = meta['rep_idx']

        if chosen_label is None:
            if meta.get('top_label_phrase') and label_fallback_strategy != 'rep_sentence_excerpt':
                chosen_label = meta['top_label_phrase']['phrase']
                label_score = float(meta['top_label_phrase']['score'])
                label_type = 'candidate'
            elif meta.get('top_label_phrase') and label_fallback_strategy == 'rep_sentence_excerpt':
                chosen_label = meta['top_label_phrase']['phrase']
                label_score = float(meta['top_label_phrase']['score'])
                label_type = 'candidate'
            elif label_fallback_strategy == 'rep_sentence_excerpt':
                rep_text = cluster_dict['sentences'][rep_idx]
                chosen_label = rep_text[:50] + '...' if len(rep_text) > 50 else rep_text
                label_score = 0.0
                label_type = 'excerpt'
            else:
                chosen_label = '' if label_fallback_strategy == 'none' else 'Unlabeled'
                label_score = None
                label_type = 'none'

        keywords = []
        for k in top_phrases[:top_keywords]:
            keywords.append({
                'phrase': k['phrase'],
                'sim': k['sim'],
                'freq_or_tfidf': k['freq_or_tfidf'],
                'score': k['score'],
            })

        labeled_clusters.append({
            'cluster_id': int(cluster_id),
            'size': len(cluster_dict['sentences']),
            'label': chosen_label,
            'label_score': float(label_score) if label_score is not None else None,
            'label_type': label_type,
            'keywords': keywords,
            'rep_id': int(rep_idx),
            'entities': meta['ents'],
            'sentence_ids': cluster_dict['indices'],
            'sentences': cluster_dict['sentences'],
            'sentences_centroid_similarity': [float(s) for s in sims.tolist()] if sims is not None and not isinstance(sims, list) else None,
            'radius': meta['radius'],
            'avg_dist': meta['avg_dist'],
            'diameter': meta['diameter'],
            'candidate_details': candidate_details,
        })
    on_process_clusters()

    return labeled_clusters, candidate_meta_by_cluster


def _build_candidates_vectorizer(
    sentences: List[str],
    tokenization_choice: str,
    max_candidates: Union[int, None],
    ngram_range: tuple,
    stop_words: list,
    max_features: Union[int, None],
    candidate_selection_strategy: str,
    tfidf_adaptive_percentile: int,
    proportional_ratio: float,
    proportional_min: Union[int, None],
    proportional_max: int,
    density_base: int,
    log_scale_factor: int,
    log_base: float,
    density_scale: int,
    density_max_dist: float,
    cluster_avg_dist: Union[float, None],
    vectorizer,
    candidate_pruning_strategy: str,
    candidate_pruning_min_df: int,
    candidate_pruning_max_df: float,
    stanza_docs: List[Document],
    is_global: bool,
):
    """Builds candidate phrases for a single cluster's sentences."""
    if not sentences:
        return {'all_phrases': [], 'candidates': [], 'col_sums': None, 'ents': [], 'token_count': 0, 'phrase_to_idx': {}}

    if tokenization_choice == 'stanza':
        allowed_upos = {'NOUN', 'PROPN', 'ADJ'}
        all_phrases_set = set()
        ents = [] if not is_global else None  # Only compute entities for local clusters
        total_tokens = 0

        for doc in stanza_docs:
            sent_id = doc.sent_id if hasattr(doc, 'sent_id') else 0  # bulk_process adds sent_id
            for ent in doc.entities:
                t = ent.text.strip().lower()
                if len(t) > 1 and ents is not None:
                    all_phrases_set.add(t)
                    ents.append({'type': ent.type, 'text': t, 'sent_id': sent_id})

            for sent in doc.sentences:
                total_tokens += len(sent.words)
                cur = []
                for w in sent.words:
                    if w.upos in allowed_upos:
                        cur.append(w.lemma if hasattr(w, 'lemma') and isinstance(w.lemma, str) else w.text)
                        max_len = max(ngram_range)
                        if len(cur) > max_len:
                            cur.pop(0)
                        for L in range(1, min(len(cur), max_len) + 1):
                            phrase = " ".join(cur[-L:]).lower()
                            if phrase not in stop_words:
                                all_phrases_set.add(phrase)
                    else:
                        cur = []

        phrases = [p.strip() for p in all_phrases_set if p.strip()]

        num_to_select = max_candidates
        if candidate_selection_strategy == 'proportional' or candidate_selection_strategy == 'adaptive':
            total_unique_candidates = len(phrases)
            proportional_count = int(total_unique_candidates * proportional_ratio)
            if proportional_min is None:
                num_to_select = min(proportional_max, proportional_count)
            else:
                num_to_select = int(max(proportional_min, min(proportional_count, proportional_max)))
        elif candidate_selection_strategy == 'density':
            avg_dist = cluster_avg_dist if cluster_avg_dist is not None else 0.2
            normalized_dist = min(avg_dist / density_max_dist, 1.0)
            num_to_select = int(density_base + density_scale * normalized_dist)
        elif candidate_selection_strategy == 'logarithmic':
            if total_tokens > 1:
                log_count = int(log_scale_factor * np.log(total_tokens) / np.log(log_base))
                num_to_select = int(max(proportional_min or 1, min(log_count, proportional_max)))
            else:
                num_to_select = proportional_min or 1
        if num_to_select is not None and len(phrases) > num_to_select:
            cnt = Counter(phrases)
            most = [p for p, _ in cnt.most_common(num_to_select)]
        else:
            most = phrases
        return {'all_phrases': phrases, 'candidates': most, 'col_sums': None, 'ents': ents, 'token_count': total_tokens, 'phrase_to_idx': {p: i for i, p in enumerate(phrases)}}

    if not vectorizer:
        if tokenization_choice == 'tfidf':
            vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words=stop_words, max_features=max_features, token_pattern=r'(?u)\b\w+\b')
        else:
            vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words, max_features=max_features)

    try:
        X = vectorizer.fit_transform(sentences)
    except ValueError:
        return {'all_phrases': [], 'candidates': [], 'col_sums': None, 'ents': None, 'token_count': 0}
    candidates = vectorizer.get_feature_names_out() if hasattr(vectorizer, 'get_feature_names_out') else vectorizer.get_feature_names()
    phrase_to_idx = {phrase: i for i, phrase in enumerate(candidates)}

    # Pruning before selection
    if candidate_pruning_strategy and X.shape[1] > 0:
        doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
        keep_mask = np.ones(X.shape[1], dtype=bool)
        if candidate_pruning_strategy == 'min_df':
            keep_mask = doc_freq >= candidate_pruning_min_df
        elif candidate_pruning_strategy == 'max_df':
            keep_mask = doc_freq / X.shape[0] <= candidate_pruning_max_df
        X = X[:, keep_mask]
        candidates = [cand for i, cand in enumerate(candidates) if keep_mask[i]]
        phrase_to_idx = {phrase: i for i, phrase in enumerate(candidates)}

        candidates = vectorizer.get_feature_names_out()
    if X.shape[1] > 0:
        col_sums = np.asarray(X.sum(axis=0)).ravel()
    else:
        col_sums = np.zeros(len(candidates))

    num_to_select = max_candidates
    token_count = int(X.sum())
    if candidate_selection_strategy == 'proportional' or (candidate_selection_strategy == 'adaptive' and tokenization_choice != 'tfidf'):
        total_unique_candidates = len(candidates)
        proportional_count = int(total_unique_candidates * proportional_ratio)
        if proportional_min is None:
            num_to_select = min(proportional_max, proportional_count)
        else:
            num_to_select = int(max(proportional_min, min(proportional_count, proportional_max)))
    elif candidate_selection_strategy == 'density':
        avg_dist = cluster_avg_dist if cluster_avg_dist is not None else 0.0
        normalized_dist = min(avg_dist / density_max_dist, 1.0)
        num_to_select = int(density_base + density_scale * normalized_dist)
    elif candidate_selection_strategy == 'logarithmic':
        if token_count > 1:
            log_count = int(log_scale_factor * np.log(token_count) / np.log(log_base))
            num_to_select = int(max(proportional_min or 1, min(log_count, proportional_max)))
        else:
            num_to_select = proportional_min or 1

    if candidate_selection_strategy == 'adaptive' and tokenization_choice == 'tfidf':
        if len(col_sums) > 2:
            sorted_scores = np.sort(col_sums)[::-1]
            normalized_scores = (sorted_scores - sorted_scores.min()) / (sorted_scores.max() - sorted_scores.min())
            line_points = np.linspace(normalized_scores[0], normalized_scores[-1], len(normalized_scores))
            distances = normalized_scores - line_points
            elbow_index = np.argmax(distances)
            if elbow_index > 0:
                threshold = sorted_scores[elbow_index]
                most = [cand for i, cand in enumerate(candidates) if col_sums[i] >= threshold]
            else:
                threshold = np.percentile(col_sums, tfidf_adaptive_percentile)
                most = [cand for i, cand in enumerate(candidates) if col_sums[i] >= threshold]
        else:
            most = list(candidates)  # Not enough data for elbow/percentile
    elif num_to_select is not None and len(candidates) > num_to_select:  # fixed_count, proportional, density, or adaptive fallback
        top_indices = np.argsort(col_sums)[::-1][:num_to_select]
        most = [candidates[i] for i in top_indices]
    elif num_to_select is not None:  # max_features might be smaller than num_to_select
        most = list(candidates)
    else:
        most = list(candidates)

    return {'all_phrases': list(candidates), 'candidates': most, 'col_sums': col_sums, 'ents': None, 'token_count': token_count, 'phrase_to_idx': phrase_to_idx}


def _top_phrases_for_cluster(
    embeddings,
    top_keywords,
    semantic_weight,
    freq_weight,
    mmr_lambda,
    lift_weight,
    global_penalty_factor,
    global_candidate_data,
    rep_sentence_strategy,
    precomputed_candidates_data,
    candidate_embeddings_map,
    predefined_keywords,
    dynamic_candidate_source,
    force_predefined_for: Union[list, None],
    thresholds: dict,
    include_candidate_details: bool = False,
    stanza_docs: List[Document] = None,
):
    if embeddings.size == 0:
        return {'rep_idx': 0, 'radius': 0, 'avg_dist': 0, 'diameter': 0, 'ents': [], 'lang': None}, [], None, [], None, None

    centroid = embeddings.mean(axis=0, keepdims=True)
    eps = 1e-12
    centroid_norm = np.linalg.norm(centroid, axis=1, keepdims=True)
    centroid = centroid / np.maximum(centroid_norm, eps)

    sims = cosine_similarity(embeddings, centroid).ravel()

    radius = float(1 - sims.min())  # furthest point from centroid
    avg_dist = float(1 - sims.mean())  # average "spread"
    pairwise = cosine_similarity(embeddings)
    pairwise = np.nan_to_num(pairwise, nan=0.0)
    diameter = float(1 - pairwise.min())

    if rep_sentence_strategy == 'centrality' and embeddings.shape[0] > 1:
        centrality_scores = pairwise.mean(axis=1)
        rep_idx = int(centrality_scores.argmax())
    else:  # 'centroid' or fallback
        rep_idx = int(sims.argmax())
    meta = {
        'rep_idx': rep_idx,
        'radius': radius,
        'avg_dist': avg_dist,
        'diameter': diameter,
    }

    all_phrases = precomputed_candidates_data.get('all_phrases', [])
    col_sums = precomputed_candidates_data.get('col_sums')
    ents = precomputed_candidates_data.get('ents', [])
    local_phrase_to_idx = precomputed_candidates_data.get('phrase_to_idx', {})
    global_phrase_to_idx = global_candidate_data.get('phrase_to_idx', {})

    meta['ents'] = ents or []

    dynamic_candidates = precomputed_candidates_data.get('candidates', [])
    predefined_candidates = list(predefined_keywords.keys()) if predefined_keywords else []

    force_predefined_labels = force_predefined_for and 'labels' in force_predefined_for
    force_predefined_keywords = force_predefined_for and 'keywords' in force_predefined_for

    label_candidates = set()
    if (not dynamic_candidate_source or 'labels' in dynamic_candidate_source) and not force_predefined_labels:
        label_candidates.update(dynamic_candidates)
    if predefined_candidates:
        label_candidates.update(predefined_candidates)

    keyword_candidates = set()
    if (not dynamic_candidate_source or 'keywords' in dynamic_candidate_source) and not force_predefined_keywords:
        keyword_candidates.update(dynamic_candidates)
    if predefined_candidates:
        keyword_candidates.update(predefined_candidates)
    if force_predefined_keywords:
        keyword_candidates = set(predefined_candidates)

    all_candidates_to_score = list(label_candidates | keyword_candidates)
    if not all_candidates_to_score:
        return meta, [], centroid, [], None, None

    cand_emb = np.array([candidate_embeddings_map[c] for c in all_candidates_to_score if c in candidate_embeddings_map])
    if cand_emb.shape[0] == 0:
        return meta, [], centroid, [], None, None

    cand_sims = cosine_similarity(cand_emb, centroid).ravel()

    if cand_sims.max() > cand_sims.min():
        sim_norm = (cand_sims - cand_sims.min()) / (cand_sims.max() - cand_sims.min())
    else:
        sim_norm = np.zeros_like(cand_sims)
    if col_sums is None or len(col_sums) == 0:
        # Frequency calculation for Stanza candidates where `col_sums` is not available.
        def count_phrase_in_tokens(phrase, token_list):
            phrase_tokens = phrase.split()
            if not phrase_tokens: return 0
            count = 0
            for i in range(len(token_list) - len(phrase_tokens) + 1):
                if token_list[i:i + len(phrase_tokens)] == phrase_tokens:
                    count += 1
            return count

        all_lemmas_or_tokens = []
        if stanza_docs:
            for doc in stanza_docs:
                for sent in doc.sentences:
                    all_lemmas_or_tokens.extend([
                        (w.lemma if hasattr(w, 'lemma') and isinstance(w.lemma, str) else w.text).lower()
                        for w in sent.words
                    ])
        freq = np.array([count_phrase_in_tokens(c, all_lemmas_or_tokens) for c in all_candidates_to_score])
        final_col_sums = freq
        if freq.max() > freq.min():
            freq_norm = (freq - freq.min()) / (freq.max() - freq.min())
        else:
            freq_norm = np.zeros_like(sim_norm)
    else:
        # TF-IDF/Count path: map local `col_sums` to the combined candidate list.
        final_freq = np.array([col_sums[local_phrase_to_idx[c]] if c in local_phrase_to_idx else 0 for c in all_candidates_to_score])
        final_col_sums = final_freq
        if final_freq.max() > final_freq.min():
            freq_norm = (final_freq - final_freq.min()) / (final_freq.max() - final_freq.min())
        else:
            freq_norm = np.zeros_like(sim_norm)

    base_score = semantic_weight * sim_norm + freq_weight * freq_norm

    lift_scores = np.zeros_like(base_score)
    if lift_weight > 0.0 and global_candidate_data.get('col_sums') is not None:
        global_sums = global_candidate_data['col_sums']
        local_total_freq = np.sum(final_col_sums)
        global_total_freq = np.sum(global_sums)

        for i, c in enumerate(all_candidates_to_score):
            if c in global_phrase_to_idx:
                local_freq_norm = (final_col_sums[i] / local_total_freq) if local_total_freq > 0 else 0
                global_freq_norm = (global_sums[global_phrase_to_idx[c]] / global_total_freq) if global_total_freq > 0 else 0
                if global_freq_norm > 0:
                    lift = local_freq_norm / global_freq_norm
                    # Use log1p to handle lift values gracefully and add to score
                    # A lift > 1 (good) gives a positive bonus, lift < 1 (bad) gives a negative penalty
                    lift_scores[i] = np.log1p(lift - 1) if lift > 0 else 0

    combined = base_score + lift_weight * lift_scores

    if global_penalty_factor < 1.0:
        for i, c in enumerate(all_candidates_to_score):
            if c in global_phrase_to_idx and c not in local_phrase_to_idx:
                combined[i] *= global_penalty_factor

    if mmr_lambda and mmr_lambda > 0.0 and len(all_candidates_to_score) > top_keywords:
        pick_idx = _mmrSelect(cand_emb, combined, top_keywords, lambda_param=mmr_lambda)  # MMR for diversity
    else:
        pick_idx = np.argsort(combined)[::-1]

    predefined_keyword_threshold = thresholds.get('predefined_keyword', 0.32)
    dynamic_keyword_threshold = thresholds.get('dynamic_keyword', 0.0)
    dynamic_label_threshold = thresholds.get('dynamic_label', 0.0)

    final_keyword_indices = []
    for i in pick_idx:
        cand_phrase = all_candidates_to_score[i]
        if cand_phrase in keyword_candidates:
            is_predefined = cand_phrase in predefined_candidates
            score = combined[i]
            if (is_predefined and score >= predefined_keyword_threshold) or \
                (not is_predefined and score >= dynamic_keyword_threshold):
                final_keyword_indices.append(i)

    top_keyword_indices = final_keyword_indices[:top_keywords]

    label_candidate_indices = [i for i, c in enumerate(all_candidates_to_score) if c in label_candidates]
    if label_candidate_indices and not force_predefined_labels:
        top_label_idx = max(label_candidate_indices, key=lambda i: combined[i])
        top_label_score = float(combined[top_label_idx])
        top_label_phrase = all_candidates_to_score[top_label_idx]
        is_predefined = top_label_phrase in predefined_candidates

        # Check if the top dynamic label meets its threshold
        if (is_predefined and top_label_score >= predefined_keyword_threshold) or \
            (not is_predefined and top_label_score >= dynamic_label_threshold):
            meta['top_label_phrase'] = {'phrase': top_label_phrase, 'score': top_label_score}

    top_phrases = []
    if top_keyword_indices:
        for i in top_keyword_indices:
            top_phrases.append({
                'phrase': all_candidates_to_score[int(i)],
                'sim': round(float(cand_sims[int(i)]), 4),
                'freq_or_tfidf': round(float(final_col_sums[int(i)]), 4) if final_col_sums is not None else None,
                'score': round(float(combined[int(i)]), 4),
            })

    candidate_details_response = None
    if include_candidate_details:
        ranked_indices = np.argsort(combined)[::-1]
        rank_map = {idx: rank for rank, idx in enumerate(ranked_indices)}

        detailed_candidates = []
        for i, cand in enumerate(all_candidates_to_score):
            detailed_candidates.append({
                'phrase': cand,
                'rank': rank_map.get(i),
                'score_final': float(combined[i]),
                'score_base': float(base_score[i]),
                'score_semantic_norm': float(sim_norm[i]),
                'score_freq_norm': float(freq_norm[i]),
                'score_lift': float(lift_scores[i]),
                'raw_semantic': float(cand_sims[i]),
                'raw_freq_or_tfidf': float(final_col_sums[i]),
                'is_predefined': cand in predefined_candidates,
                'is_local': cand in local_phrase_to_idx,
                'is_global': cand in global_phrase_to_idx,
            })
        candidate_details_response = {
            'initial_candidate_count': len(all_phrases),
            'final_candidate_count': len(all_candidates_to_score),
            'label_candidate_count': len(label_candidates),
            'keyword_candidate_count': len(keyword_candidates),
            'candidates': sorted(detailed_candidates, key=lambda x: x['rank'])
        }

    return meta, top_phrases, centroid, sims, (all_candidates_to_score, cand_emb, combined, all_phrases), candidate_details_response


def _mmrSelect(candidates_emb, candidate_scores, k, lambda_param=0.5):
    """Selects k diverse items from a list of candidates using Maximal Marginal Relevance."""
    n = candidates_emb.shape[0]
    if n == 0:
        return []
    selected = []
    idxs = np.argsort(candidate_scores)[::-1]
    if len(idxs) == 0:
        return []
    first = idxs[0]
    selected.append(first)
    while len(selected) < k and len(selected) < n:
        remaining = [i for i in idxs if i not in selected]
        best = None
        best_score = -1e9
        for r in remaining:
            sim_to_centroid = candidate_scores[r]  # Relevance
            sim_selected = max(cosine_similarity(
                candidates_emb[r:r + 1], candidates_emb[selected]).ravel()) if selected else 0.0
            mmr_score = lambda_param * sim_to_centroid - (1.0 - lambda_param) * sim_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best = r
        if best is None:
            break
        selected.append(best)
    return selected
