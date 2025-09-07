# disclaimer: this file was written by AI, all of it.
# I felt this needs a disclaimer here.
import base64
import io
import logging
from typing import List, Union, Dict, Any

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, OPTICS
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from stanza import Document
from wordcloud import WordCloud
from sklearn.neighbors import KernelDensity
from baistro.model_control.infer_result import InferTracker
from baistro.model_control.models import models
from baistro.models.vector_text import VectorTextModel
import pandas as pd

_plot_options = [
    "cluster-scatter-pca",
    "cluster-scatter-tsne",
    "cluster-density",
    "cluster-heatmap",
    "cluster-sizes",
    "embedding-histogram",
    "silhouette",
    "wordcloud-combined",
    "wordcloud",
    "wordcloud-overview",
    "graph",
    "cluster-similarity",
    "cluster-separation",
    "keyword-cooccurrence-graph",
    "keyword-cooccurrence-heatmap",
    "cluster-overlap-graph",
    "treemap",
    "dendrogram-heatmap",
    "cluster-distance-distribution",
    "keyword-contribution",
    "bipartite-cluster-sentence-graph",
    "characteristic-keywords",
    "word-cooccurrence-graph",
    "word-cooccurrence-heatmap",
    "label-word-semantic-hierarchy",
    "cluster-cohesion-plot",
    "cluster-outlier-scores",
    "sentence-similarity-distribution",
    "cluster-label-similarity-heatmap",
    "keyword-rarity-plot",
    "cluster-keyword-bipartite-graph",
    "topic-world-map",
    "semantic-gravity-well",
    "cluster-fingerprint-radar",
    "corpus-topic-radar",
    "keyword-trajectory",
    "sentence-bridge-identifier",
    "galaxy-map-constellations",
    "gravitational-force-network",
    "cluster-aura-plot",
]


def make_cluster_plots(
    plots: List[str],
    corpus_embeddings: np.ndarray,
    cluster_assignment: np.ndarray,
    labeled_clusters: List[dict],
    clusters: Dict[Any, Any],
    tracker,
    model: VectorTextModel,
    clustering_model: Union[AgglomerativeClustering, OPTICS],
    vectorizer: TfidfVectorizer,
    tfidf_matrix: np.ndarray,
    candidate_meta_by_cluster: dict,
    corpus: List[str],
    stanza_docs: List[Document],
    options: dict,
):
    n_clusters = len(set(cluster_assignment))

    assets = []

    def plot_word_cloud(only_overview=False, only_combined=False):
        wordclouds = make_topic_wordclouds(
            labeled_clusters, candidate_meta_by_cluster, stanza_docs,
            only_overview=only_overview,
            only_combined=only_combined,
        )
        for cid, uri in wordclouds.items():
            assets.append({"type": cid, "value": uri})

    _plotter = {
        'cluster-scatter-pca': lambda: make_cluster_scatter(corpus_embeddings, cluster_assignment),
        'cluster-scatter-tsne': lambda: make_cluster_scatter(corpus_embeddings, cluster_assignment, method='tsne'),
        'cluster-density': lambda: make_cluster_density(corpus_embeddings, cluster_assignment),
        'cluster-heatmap': lambda: make_cluster_heatmap(corpus_embeddings),
        'cluster-sizes': lambda: make_cluster_size_bar(cluster_assignment),
        'embedding-histogram': lambda: make_embedding_histogram(corpus_embeddings),
        'silhouette': lambda: make_silhouette_plot(corpus_embeddings, cluster_assignment) if 2 < n_clusters <= len(corpus_embeddings) - 1 else None,
        'wordcloud': lambda: plot_word_cloud(),
        'wordcloud-overview': lambda: plot_word_cloud(only_overview=True),
        'wordcloud-combined': lambda: plot_word_cloud(only_combined=True),
        'graph': lambda: make_cluster_graph(labeled_clusters, model),
        'cluster-similarity': lambda: make_cluster_similarity(cluster_assignment, len(set(cluster_assignment))) if n_clusters > 1 else None,
        'cluster-separation': lambda: make_cluster_separation(corpus_embeddings, cluster_assignment, len(set(cluster_assignment))) if n_clusters > 1 else None,
        'keyword-cooccurrence-graph': lambda: make_keyword_cooccurrence_graph(labeled_clusters),
        'keyword-cooccurrence-heatmap': lambda: make_keyword_cooccurrence_heatmap(labeled_clusters),
        'cluster-overlap-graph': lambda: make_cluster_overlap_graph(labeled_clusters, corpus_embeddings, cluster_assignment) if n_clusters > 1 else None,
        'bipartite-cluster-sentence-graph': lambda: make_bipartite_cluster_sentence_graph(labeled_clusters, corpus_embeddings, cluster_assignment) if n_clusters > 1 else None,
        'treemap': lambda: make_treemap_plot(clustering_model, labeled_clusters),
        'dendrogram-heatmap': lambda: make_dendrogram_heatmap(clustering_model, corpus_embeddings),
        'cluster-distance-distribution': lambda: make_cluster_distance_distribution_plot(corpus_embeddings, cluster_assignment) if n_clusters > 1 else None,
        'keyword-contribution': lambda: make_keyword_contribution_plot(labeled_clusters, vectorizer, tfidf_matrix),
        'characteristic-keywords': lambda: make_characteristic_keywords_plot(labeled_clusters, clusters, corpus, vectorizer, tfidf_matrix),
        'word-cooccurrence-graph': lambda: make_word_cooccurrence_graph(labeled_clusters, candidate_meta_by_cluster),
        'word-cooccurrence-heatmap': lambda: make_word_cooccurrence_heatmap(labeled_clusters, candidate_meta_by_cluster),
        'label-word-semantic-hierarchy': lambda: make_label_word_semantic_hierarchy(labeled_clusters, candidate_meta_by_cluster),
        'cluster-cohesion-plot': lambda: make_cluster_cohesion_plot(labeled_clusters),
        'cluster-outlier-scores': lambda: make_cluster_outlier_scores(labeled_clusters, corpus_embeddings, cluster_assignment),
        'sentence-similarity-distribution': lambda: make_sentence_similarity_distribution(corpus_embeddings),
        'cluster-label-similarity-heatmap': lambda: make_cluster_label_similarity_heatmap(labeled_clusters, options.get('keywords', {})),
        'keyword-rarity-plot': lambda: make_keyword_rarity_plot(labeled_clusters, candidate_meta_by_cluster, corpus),
        'cluster-keyword-bipartite-graph': lambda: make_cluster_keyword_bipartite_graph(labeled_clusters),
        'topic-world-map': lambda: make_topic_world_map(corpus_embeddings, labeled_clusters, cluster_assignment),
        'semantic-gravity-well': lambda: make_semantic_gravity_well_plot(corpus_embeddings, labeled_clusters),
        'cluster-fingerprint-radar': lambda: make_cluster_fingerprint_radar_chart(labeled_clusters, candidate_meta_by_cluster),
        'corpus-topic-radar': lambda: make_corpus_topic_radar(corpus_embeddings, labeled_clusters, candidate_meta_by_cluster),
        'keyword-trajectory': lambda: make_keyword_trajectory_plot(labeled_clusters),
        'sentence-bridge-identifier': lambda: make_sentence_bridge_identifier_plot(corpus_embeddings, labeled_clusters, candidate_meta_by_cluster),
        'galaxy-map-constellations': lambda: make_galaxy_map_plot(corpus_embeddings, labeled_clusters, cluster_assignment),
        'gravitational-force-network': lambda: make_gravitational_force_network(labeled_clusters, corpus_embeddings, cluster_assignment),
        'cluster-aura-plot': lambda: make_cluster_aura_plot(corpus_embeddings, labeled_clusters, cluster_assignment, candidate_meta_by_cluster),
    }

    for plot in plots:
        on_plotted_diagram = tracker(f'plot_diagrams__{plot}')
        if plot in _plotter:
            logging.info(f"Start generating plot '{plot}'...")
            try:
                img_res = _plotter[plot]()
                if img_res:
                    assets.append({"type": plot, "value": img_res})
                elif plot != 'wordcloud' and plot != 'wordcloud-overview' and plot != 'wordcloud-combined':
                    logging.warning(f"Plot type '{plot}' returned no image or was skipped.")
            except Exception as e:
                logging.exception(f"Error generating plot '{plot}': {e}")
        else:
            logging.warning(f"Plot type '{plot}' not recognized or not yet implemented.")
        on_plotted_diagram()

    return assets


def _plot_to_data_uri():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


def make_topic_world_map(embeddings, labeled_clusters, labels, method="pca"):
    """A 2D density contour plot where clusters are represented as "continents" or "islands" in a semantic ocean.

    **Details**:
    This visualization shows clusters as continents/islands in a semantic ocean.
    """
    if embeddings.shape[1] > 2:
        if method == "tsne":
            reducer = TSNE(
                n_components=2, random_state=42,
                perplexity=min(30.0, len(embeddings) - 1),
            )
        else:
            reducer = PCA(n_components=2)
        coords = reducer.fit_transform(embeddings)
    else:
        coords = embeddings

    # 1. Create the density contour map (the "world")
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    cbar_ax = None
    try:
        import seaborn as sns
        # Create a mappable object from the KDE plot to use for the color bar
        kde_plot = sns.kdeplot(x=coords[:, 0], y=coords[:, 1], cmap="Blues", fill=True, thresh=0.05, levels=10, ax=ax, cbar=True, cbar_ax=None)
        cbar = kde_plot.collections[0].colorbar
        cbar.set_label('Semantic Density', rotation=270, labelpad=15)
    except ImportError:
        # Fallback if seaborn is not available
        kde = KernelDensity(bandwidth=0.5).fit(coords)  # type: ignore
        x_grid, y_grid = np.meshgrid(
            np.linspace(coords[:, 0].min() - 1, coords[:, 0].max() + 1, 150),
            np.linspace(coords[:, 1].min() - 1, coords[:, 1].max() + 1, 150),
        )
        grid = np.c_[x_grid.ravel(), y_grid.ravel()]
        z = np.exp(kde.score_samples(grid)).reshape(x_grid.shape)
        ax.contourf(x_grid, y_grid, z, levels=10, cmap="Blues", alpha=0.8)

    # 2. Plot cluster centroids as "mountain peaks" and label them
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for i, lc in enumerate(labeled_clusters):
        cluster_indices = lc['sentence_ids']
        cluster_coords = coords[cluster_indices]
        centroid_coord = cluster_coords.mean(axis=0)
        color = colors(i % len(unique_labels))
        ax.scatter(centroid_coord[0], centroid_coord[1], color=color, s=150, ec='black', marker='X', zorder=10)
        ax.text(centroid_coord[0], centroid_coord[1] + 0.1, f"C{lc['cluster_id']}: {lc['label']}", fontsize=10, weight='bold', ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

    plt.title("Topic World Map", fontsize=16)
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    return _plot_to_data_uri()


def make_semantic_gravity_well_plot(embeddings: np.ndarray, labeled_clusters: List[dict], k: int = 15, method: str = "pca"):
    """A 3D surface plot where the Z-axis represents semantic density, creating "gravity wells" around dense topics.

    **Details**:
    """
    if len(embeddings) < k + 1:
        logging.warning(f"Not enough samples for gravity well plot (need > {k}).")
        return None

    # 1. Reduce dimensionality for the X, Y plane
    if embeddings.shape[1] > 2:
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(embeddings)
    else:
        coords = embeddings

    # 2. Calculate semantic "gravity" for the Z axis
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    # Avg distance to k neighbors (excluding self)
    avg_dist_to_neighbors = distances[:, 1:].mean(axis=1)
    # Gravity is inverse distance (add epsilon for stability)
    gravity = 1.0 / (avg_dist_to_neighbors + 1e-9)

    # 3. Create a grid for the surface plot
    from scipy.interpolate import griddata
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

    # Interpolate the gravity values onto the grid
    grid_z = griddata(coords, gravity, (grid_x, grid_y), method='cubic', fill_value=0)

    # 4. Plot the 3D surface
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none', alpha=0.8)
    # Add a color bar to explain the Z-axis (gravity)
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1, label="Semantic Gravity (Density)")

    # 5. Overlay cluster centroids as markers
    colors = plt.cm.get_cmap('tab10', len(labeled_clusters))
    for i, lc in enumerate(labeled_clusters):
        cluster_indices = lc['sentence_ids']
        cluster_coords = coords[cluster_indices]
        centroid_coord_2d = cluster_coords.mean(axis=0) if len(cluster_coords) > 0 else np.array([0, 0])
        # Interpolate the Z value for the centroid
        centroid_z = griddata(coords, gravity, (centroid_coord_2d[0], centroid_coord_2d[1]), method='nearest') if len(coords) > 0 else 0
        ax.scatter(centroid_coord_2d[0], centroid_coord_2d[1], centroid_z,
                   marker='o', s=100, c=[colors(i)], ec='white', depthshade=True, zorder=10)
        ax.text(centroid_coord_2d[0], centroid_coord_2d[1], centroid_z * 1.1,
                f"C{lc['cluster_id']}: {lc['label']}", color='white', fontsize=9, ha='center',
                zorder=11,  # Set a higher zorder for the text to ensure it's on top
                bbox=dict(boxstyle='round,pad=0.2', fc='black', ec='none', alpha=0.5))

    ax.set_title("Semantic 'Gravity Well' Map", fontsize=18)
    ax.set_xlabel(f"{method.upper()} Dimension 1")
    ax.set_ylabel(f"{method.upper()} Dimension 2")
    ax.set_zlabel("Semantic Gravity (Density)")
    ax.view_init(elev=50, azim=-65)  # Set a good viewing angle

    return _plot_to_data_uri()


def make_cluster_fingerprint_radar_chart(labeled_clusters: List[dict], candidate_meta_by_cluster: dict):
    """A radar chart comparing clusters across normalized metrics like Size, Spread, and Keyword Specificity.

    **Details**:
    """
    if not labeled_clusters:
        return None

    metrics = ['Size', 'Spread', 'Diameter', 'Keyword Specificity']
    data = []

    for lc in labeled_clusters:
        cid = lc['cluster_id']
        # Calculate Keyword Specificity
        specificity = 0
        if cid in candidate_meta_by_cluster:
            candidates, _, scores, _ = candidate_meta_by_cluster[cid]
            if len(scores) > 0:
                # Specificity can be the average score of top keywords
                top_indices = np.argsort(scores)[-5:]
                specificity = scores[top_indices].mean()

        data.append({
            'group': f"C{lc['cluster_id']}: {lc['label']}",
            'Size': lc['size'],
            'Spread': lc['avg_dist'],
            'Diameter': lc['diameter'],
            'Keyword Specificity': specificity
        })

    df = pd.DataFrame(data)

    # Normalize each metric to a 0-1 scale for the radar chart
    for metric in metrics:
        min_val, max_val = df[metric].min(), df[metric].max()
        if max_val - min_val > 0:
            df[metric] = (df[metric] - min_val) / (max_val - min_val)
        else:
            df[metric] = 0.5  # Assign a neutral value if all are the same

    # Plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_rlabel_position(30)

    colors = plt.cm.get_cmap('tab10', len(df))
    for i, row in df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, color=colors(i), linewidth=2, linestyle='solid', label=row['group'])
        # Add percentage labels at each vertex
        for angle, value in zip(angles[:-1], values[:-1]):
            ax.text(angle, value + 0.08, f"{value:.0%}",
                    ha='center', va='center', fontsize=9, color=colors(i),
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))

        ax.fill(angles, values, color=colors(i), alpha=0.2)

    plt.title('Cluster Fingerprint Radar', size=20, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    return _plot_to_data_uri()


def make_corpus_topic_radar(corpus_embeddings: np.ndarray, labeled_clusters: List[dict], candidate_meta_by_cluster: dict):
    """A radar chart showing the entire corpus's overall affinity or similarity to each identified topic.

    **Details**:
    """
    if not labeled_clusters: return None

    labels = []
    topic_embeddings = []
    for lc in labeled_clusters:
        cid = lc['cluster_id']
        if cid in candidate_meta_by_cluster:
            _, cand_emb, _, _ = candidate_meta_by_cluster[cid]
            if cand_emb is not None and len(cand_emb) > 0:
                # Use the mean of candidate embeddings as the topic embedding
                topic_emb = cand_emb.mean(axis=0, keepdims=True)
                topic_embeddings.append(topic_emb / np.linalg.norm(topic_emb))
                labels.append(f"C{lc['cluster_id']}: {lc['label']}")

    if not topic_embeddings: return None

    topic_embeddings = np.vstack(topic_embeddings)
    # Calculate similarity of each corpus sentence to each topic
    sim_matrix = cosine_similarity(corpus_embeddings, topic_embeddings)

    # For each topic, find the average similarity score from all corpus sentences
    corpus_affinity = sim_matrix.mean(axis=0)

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    values_list = corpus_affinity.tolist()
    values = np.array(values_list + values_list[:1])

    # Add percentage labels at each vertex
    for angle, value in zip(angles[:-1], values[:-1]):
        ax.text(angle, value + 0.02, f"{value:.1%}",
                ha='center', va='bottom', fontsize=9, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.3)

    plt.title('Corpus Thematic Composition', size=16, y=1.1)
    return _plot_to_data_uri()


def make_keyword_trajectory_plot(labeled_clusters: List[dict]):
    """A line chart for each cluster, showing the "flow" or presence of its top keywords across the sequence of sentences within it.

    **Details**:
    This uses a multi-line chart of rolling averages to show the "flow" of the topic.
    """
    import seaborn as sns
    if not labeled_clusters: return None

    # Determine grid size
    n_clusters = len(labeled_clusters)
    cols = int(np.ceil(np.sqrt(n_clusters)))
    rows = int(np.ceil(n_clusters / cols)) if cols > 0 else 0
    if rows == 0 or cols == 0: return None

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5), squeeze=False, sharex=True)
    axes = axes.flatten()

    for i, lc in enumerate(labeled_clusters):
        ax = axes[i]
        keywords = [kw['phrase'] for kw in lc['keywords']]
        sentences = lc['sentences']
        if not keywords or len(sentences) < 2:
            ax.axis('off')
            if keywords:
                ax.set_title(f"C{lc['cluster_id']}: {lc['label']} (Not enough data)")
            continue

        # Create a presence matrix (sentences x keywords)
        presence_matrix = np.zeros((len(sentences), len(keywords)))
        for sent_idx, sentence in enumerate(sentences):
            for kw_idx, keyword in enumerate(keywords):
                if keyword in sentence.lower():
                    presence_matrix[sent_idx, kw_idx] = 1

        # Calculate rolling average to smooth the trajectory
        window_size = max(1, len(sentences) // 10)
        presence_df = pd.DataFrame(presence_matrix, columns=keywords)
        smoothed_df = presence_df.rolling(window=window_size, center=True, min_periods=1).mean()

        # Plot each keyword as a line
        palette = sns.color_palette("viridis", n_colors=len(keywords))
        for j, keyword in enumerate(keywords):
            ax.plot(smoothed_df.index, smoothed_df[keyword], label=keyword, color=palette[j], linewidth=2)

        ax.set_title(f"C{lc['cluster_id']}: {lc['label']}", fontsize=12)
        ax.set_ylabel("Smoothed Presence")
        ax.set_xlabel("Sentence Sequence")
        ax.legend(fontsize='small')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    for i in range(n_clusters, len(axes)):
        axes[i].axis('off')

    plt.suptitle("Keyword Trajectory Line Charts", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return _plot_to_data_uri()


def make_cluster_density(embeddings, labels, method="pca"):
    """A 2D density map (KDE) showing where sentences are most concentrated, with cluster points overlaid.

    **Details**:
    This plot helps visualize the "hotspots" in the semantic space.
    """
    reducer = PCA(n_components=2)
    coords = reducer.fit_transform(embeddings)
    kde = KernelDensity(bandwidth=0.5).fit(coords)
    x, y = np.meshgrid(
        np.linspace(coords[:, 0].min() - 1, coords[:, 0].max() + 1, 200),
        np.linspace(coords[:, 1].min() - 1, coords[:, 1].max() + 1, 200),
    )
    grid = np.c_[x.ravel(), y.ravel()]
    z = np.exp(kde.score_samples(grid)).reshape(x.shape)
    plt.figure(figsize=(6, 6))

    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    plt.contourf(x, y, z, levels=30, cmap="Blues", alpha=0.6)

    for i, label in enumerate(unique_labels):
        cluster_coords = coords[labels == label]
        plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], color=colors(i), s=15, label=f'Cluster {label}')

    plt.title("Cluster density (KDE over PCA projection)")
    plt.legend()
    return _plot_to_data_uri()


def make_sentence_bridge_identifier_plot(embeddings: np.ndarray, labeled_clusters: List[dict], candidate_meta_by_cluster: dict, top_n: int = 15):
    """A bar chart highlighting sentences that are semantically "between" two or more clusters, acting as bridges.

    **Details**:
    This plot is useful for identifying transitional or ambiguous sentences that connect different topics.
    """
    if len(labeled_clusters) < 2: return None

    import seaborn as sns

    # We need the full corpus text to display sentences
    corpus = [sent for lc in labeled_clusters for sent in lc['sentences']]

    # 1. Get cluster centroids
    centroids = []
    cluster_ids = []
    for lc in labeled_clusters:
        # Use the cluster's actual sentence embeddings for a more accurate centroid
        cluster_embeddings = embeddings[lc['sentence_ids']]
        if len(cluster_embeddings) > 0:
            centroid = cluster_embeddings.mean(axis=0, keepdims=True)
            centroids.append(centroid / np.linalg.norm(centroid))
            cluster_ids.append(lc['cluster_id'])

    if not centroids: return None
    centroids = np.vstack(centroids)
    # Map original cluster IDs to their index in the `centroids` array
    cid_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}

    # 2. Calculate similarity of all sentences to all centroids
    sim_matrix = cosine_similarity(embeddings, centroids)

    # 3. Find "bridge" sentences
    bridge_scores = []
    for i in range(len(embeddings)):
        sorted_sims = np.sort(sim_matrix[i, :])[::-1]
        # A good bridge sentence has high similarity to at least two clusters,
        # and the scores for those two are close.
        if len(sorted_sims) > 1:
            score = (sorted_sims[0] + sorted_sims[1]) / 2 - (sorted_sims[0] - sorted_sims[1])
            # Find the original cluster IDs for the top 2 similarities
            top_indices = np.argsort(sim_matrix[i, :])[::-1]
            c1_idx, c2_idx = top_indices[0], top_indices[1]
            c1_id, c2_id = cluster_ids[c1_idx], cluster_ids[c2_idx]
            bridge_scores.append((i, score, c1_id, c2_id))

    # 4. Get top N bridge sentences
    top_bridges = sorted(bridge_scores, key=lambda x: x[1], reverse=True)[:top_n]
    if not top_bridges: return None

    # Create more informative labels, including the truncated sentence text
    labels = []
    for b in top_bridges:
        sent_text = corpus[b[0]].replace('\n', ' ').strip()
        labels.append(f"S{b[0]} (C{b[2]}↔C{b[3]}): {sent_text[:70]}...")
    scores = [b[1] for b in top_bridges]

    plt.figure(figsize=(12, max(6, top_n * 0.5)))
    ax = sns.barplot(x=scores, y=labels, hue=labels, orient='h', palette='viridis', legend=False)
    ax.set_title(f"Top {top_n} Semantic Bridge Sentences", fontsize=16)
    ax.set_xlabel("Bridge Score (Higher is a better bridge)")
    ax.set_ylabel("Sentence ID")
    plt.tight_layout()
    return _plot_to_data_uri()


def make_galaxy_map_plot(embeddings, labeled_clusters, labels, method="pca"):
    """A visually rich plot combining a density "nebula" with a network "constellation" of cluster centroids.

    **Details**:
    """
    if len(labeled_clusters) < 2: return None

    # 1. Create 2D coordinates for all sentences
    if embeddings.shape[1] > 2:
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(embeddings)
    else:
        coords = embeddings

    fig, ax = plt.subplots(figsize=(16, 12))

    # 2. Draw the density "Nebula" background
    import seaborn as sns
    kde_plot = sns.kdeplot(x=coords[:, 0], y=coords[:, 1], cmap="mako", fill=True, thresh=0.05, levels=15, ax=ax, alpha=1.0, cbar=True)
    cbar = kde_plot.collections[0].colorbar
    cbar.set_label('Semantic Density', rotation=270, labelpad=15)

    # 3. Calculate cluster centroids and their similarity
    unique_labels = sorted(np.unique(labels))
    centroids_2d = np.array([coords[labels == label].mean(axis=0) for label in unique_labels])
    centroids_high_d = np.array([embeddings[labels == label].mean(axis=0) for label in unique_labels])
    sim_matrix = cosine_similarity(centroids_high_d)

    # 4. Build and draw the "Constellation" network
    G = nx.Graph()
    # Use a more vibrant color palette for better contrast
    colors = plt.cm.get_cmap('viridis', len(labeled_clusters))
    node_sizes = []
    node_labels = {}

    for i, lc in enumerate(labeled_clusters):
        cid = lc['cluster_id']
        label_idx = unique_labels.index(cid)
        G.add_node(cid, pos=centroids_2d[label_idx])
        node_sizes.append(lc['size'] * 20 + 50)  # Scale node size by cluster size
        node_labels[cid] = f"C{cid}\n{lc['label']}"

    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            similarity = sim_matrix[i, j]
            if similarity > 0.4:  # Similarity threshold for drawing a link
                G.add_edge(unique_labels[i], unique_labels[j], weight=similarity)

    pos = nx.get_node_attributes(G, 'pos')
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=[colors(i) for i in range(len(labeled_clusters))],
                           edgecolors='white', linewidths=1.5, alpha=1.0, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='white', alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_color='black', font_weight='bold', ax=ax,
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.6))

    ax.set_title("Topic Galaxy Map: Constellations and Nebulae", fontsize=18)
    ax.set_xlabel(f"{method.upper()} Dimension 1")
    ax.set_ylabel(f"{method.upper()} Dimension 2")
    ax.grid(True, linestyle='--', alpha=0.1)
    return _plot_to_data_uri()


def make_gravitational_force_network(labeled_clusters, embeddings, labels, title="Gravitational Force-Directed Network"):
    """A force-directed network where a node's "mass" (based on size and cohesion) influences the final layout.

    **Details**:
    """
    if len(labeled_clusters) < 2: return None

    G = nx.Graph()
    colors = plt.cm.get_cmap('tab10', len(labeled_clusters))

    # 1. Calculate centroids and similarity matrix
    unique_labels = sorted(np.unique(labels))
    centroids = np.array([embeddings[labels == label].mean(axis=0) for label in unique_labels])
    sim_matrix = cosine_similarity(centroids)

    # 2. Add nodes with "mass" attribute
    for i, lc in enumerate(labeled_clusters):
        # Mass = size * cohesion (inverse of spread). Tighter clusters are more massive.
        mass = lc['size'] * (1.0 / (lc['avg_dist'] + 1e-6))
        G.add_node(lc['cluster_id'], label=f"C{lc['cluster_id']}\n{lc['label']}",
                   size=lc['size'], color=colors(i), mass=mass)

    # 3. Add edges based on similarity
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            similarity = sim_matrix[i, j]
            if similarity > 0.1:  # Low threshold to allow layout physics to work
                # Spring strength is proportional to similarity
                G.add_edge(unique_labels[i], unique_labels[j], weight=similarity)

    # 4. Create a custom force-directed layout
    # The 'k' parameter is adjusted by node mass, pulling massive nodes closer.
    pos = nx.spring_layout(G, k=1.5 / np.sqrt(len(G.nodes())), iterations=100, weight='weight', seed=42)

    plt.figure(figsize=(14, 12))
    node_sizes = [G.nodes[n]['mass'] * 5 for n in G.nodes()]  # Visual size based on mass
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    node_labels = {n: G.nodes[n]['label'] for n in G.nodes()}
    edge_widths = [G[u][v]['weight'] * 4 for u, v in G.edges()]

    nx.draw(G, pos, with_labels=False, node_size=node_sizes, node_color=node_colors,
            width=edge_widths, edge_color='#cccccc', alpha=0.8)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black')

    plt.title(title, fontsize=18)
    return _plot_to_data_uri()


def make_cluster_aura_plot(embeddings, labeled_clusters, labels, candidate_meta_by_cluster, method="pca"):
    """A 2D plot where each cluster is represented by an "aura" of concentric circles encoding its size, spread, and keyword specificity.

    **Details**:
    encoding its size, spread, and keyword specificity.
    """
    if len(labeled_clusters) < 1: return None

    # 1. Get 2D coordinates of cluster centroids
    if embeddings.shape[1] > 2:
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(embeddings)
    else:
        coords = embeddings

    unique_labels = sorted(np.unique(labels))
    centroids_2d = np.array([coords[labels == label].mean(axis=0) for label in unique_labels])

    # 2. Gather metrics for the auras
    metrics = []
    for lc in labeled_clusters:
        cid = lc['cluster_id']
        specificity = 0
        if cid in candidate_meta_by_cluster:
            _, _, scores, _ = candidate_meta_by_cluster.get(cid, (None, None, [], None))
            if len(scores) > 0:
                specificity = np.mean(np.sort(scores)[-5:])

        metrics.append({
            'size': lc['size'],
            'spread': lc['avg_dist'],
            'specificity': specificity,
            'label': f"C{cid}: {lc['label']}"
        })

    df = pd.DataFrame(metrics)
    # Normalize metrics for visual mapping
    df['size_norm'] = (df['size'] - df['size'].min()) / (df['size'].max() - df['size'].min() + 1e-9)
    df['spread_norm'] = (df['spread'] - df['spread'].min()) / (df['spread'].max() - df['spread'].min() + 1e-9)
    df['spec_norm'] = (df['specificity'] - df['specificity'].min()) / (df['specificity'].max() - df['specificity'].min() + 1e-9)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(14, 12))
    colors = plt.cm.get_cmap('tab10', len(labeled_clusters))

    for i, row in df.iterrows():
        x, y = centroids_2d[i]
        color = colors(i)

        # Outermost Aura: Keyword Specificity (color alpha)
        ax.add_artist(plt.Circle((x, y), 0.15, color=color, alpha=0.1 + row['spec_norm'] * 0.4, zorder=1))

        # Middle Aura: Spread (radius)
        ax.add_artist(plt.Circle((x, y), 0.05 + row['spread_norm'] * 0.08, color=color, alpha=0.5, zorder=2))

        # Innermost Core: Size (radius)
        ax.add_artist(plt.Circle((x, y), 0.02 + row['size_norm'] * 0.05, color=color, alpha=1.0, zorder=3))

        # Label
        label_text = f"{row['label']}\n" \
                     f"Size: {row['size']}\n" \
                     f"Spread: {row['spread']:.2f}\n" \
                     f"Spec: {row['specificity']:.2f}"

        ax.text(x, y + 0.18, label_text, ha='center', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

    # Auto-adjust plot limits
    ax.set_xlim(centroids_2d[:, 0].min() - 0.5, centroids_2d[:, 0].max() + 0.5)
    ax.set_ylim(centroids_2d[:, 1].min() - 0.5, centroids_2d[:, 1].max() + 0.5)
    ax.set_aspect('equal', adjustable='box')

    ax.set_title("Cluster Aura Map", fontsize=18)
    ax.set_xlabel(f"{method.upper()} Dimension 1")
    ax.set_ylabel(f"{method.upper()} Dimension 2")
    ax.grid(True, linestyle='--', alpha=0.3)

    # Add a descriptive legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', label='Core Radius ~ Cluster Size', markersize=8, ls=''),
        Line2D([0], [0], marker='o', color='gray', label='Middle Aura Radius ~ Spread/Avg. Distance', markersize=12, alpha=0.5, ls=''),
        Line2D([0], [0], marker='o', color='gray', label='Outer Aura Brightness ~ Keyword Specificity', markersize=16, alpha=0.2, ls='')
    ]
    ax.legend(handles=legend_elements, loc='best', title="Aura Key")

    return _plot_to_data_uri()


def make_cluster_heatmap(embeddings):
    """A heatmap of the raw sentence-to-sentence similarity matrix.

    **Details**:
    This provides a low-level view of the relationships in the data before clustering.
    """
    sim_matrix = np.inner(embeddings, embeddings)
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="cosine similarity")
    plt.title("Embedding similarity heatmap")
    return _plot_to_data_uri()


def make_cluster_size_bar(labels):
    """A simple bar chart showing the number of sentences in each cluster.

    **Details**:
    Quickly see the distribution of sentences across the identified clusters.
    """
    unique, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.get_cmap('tab10', len(unique))
    x_values = unique.astype(int)
    plt.figure(figsize=(6, 4))
    bars = plt.bar(x_values, counts, color=[colors(i) for i in range(len(unique))])
    plt.title("Cluster sizes")
    plt.xlabel("Cluster ID")
    plt.ylabel("Count")
    plt.xticks(x_values)
    plt.bar_label(bars, labels=counts, label_type='edge')
    return _plot_to_data_uri()


def make_silhouette_plot(embeddings, labels):
    """A silhouette plot for each cluster, which helps visualize how well-separated and dense the clusters are.

    **Details**:
    The overall silhouette score is also shown. Scores closer to 1 are better.
    """
    score = silhouette_score(embeddings, labels, metric="cosine")
    values = silhouette_samples(embeddings, labels, metric="cosine")
    y_lower = 10
    plt.figure(figsize=(7, 5))
    for i in np.unique(labels):
        ith_values = values[labels == i]
        ith_values.sort()
        size_i = ith_values.shape[0]
        y_upper = y_lower + size_i
        color = plt.cm.tab10(i / max(1, len(set(labels)) - 1))
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_values, facecolor=color, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10
    plt.axvline(x=score, color="red", linestyle="--")
    plt.title(f"Silhouette plot (score={score:.2f})")
    plt.xlabel("Silhouette coefficient")
    return _plot_to_data_uri()


def make_embedding_histogram(embeddings):
    """A histogram of the raw values in the sentence embeddings.

    **Details**:
    Useful for diagnosing embedding issues like normalization or scaling problems.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(embeddings.ravel(), bins=50, color="steelblue", alpha=0.7)
    plt.title("Embedding value distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    return _plot_to_data_uri()


def make_topic_wordclouds(labeled_clusters, candidate_meta_by_cluster, all_stanza_docs: List[Document], only_overview=False, only_combined=False):
    """Generates word clouds for clusters.

    **Details**:
    - `wordcloud`: Generates both the overview grid and a separate, larger word cloud for each cluster.
    - `wordcloud-overview`: An overview grid showing a small word cloud for each individual cluster.
    - `wordcloud-combined`: A single word cloud generated from the text of all clusters combined.

    This function uses the raw words from the sentences in each cluster to generate frequencies. It's a simple way to visualize the most frequent terms.
    """
    uris = {}
    all_freqs = Counter()
    cluster_wordclouds_data = []

    # A simple list of common stop words to filter out noise from the raw text.
    stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the', 'their', 'then', 'this', 'to', 'was', 'will', 'with', 'i', 'you', 'he', 'she', 'we', 'they'}

    for cluster in labeled_clusters:  # Generate individual word clouds
        cid = cluster['cluster_id']
        cluster_indices = cluster['sentence_ids']
        cluster_docs = [all_stanza_docs[i] for i in cluster_indices]

        if not cluster_docs:
            continue

        # Generate frequencies from the actual words in the cluster's sentences
        words = []
        for doc in cluster_docs:
            for sent in doc.sentences:
                for word in sent.words:
                    w_t = word.text
                    # note: lemma is done on cluster sentences, not on initial tokenization
                    if hasattr(word, 'lemma') and isinstance(word.lemma, str):
                        w_t = word.lemma
                    w_t = w_t.lower()
                    if w_t not in stop_words and len(w_t) > 1:
                        words.append(w_t)

        freqs = Counter(words)

        if not freqs:
            continue

        if only_combined:
            all_freqs.update(freqs)

        wc = WordCloud(width=400, height=300, background_color="white").generate_from_frequencies(freqs)

        # Add keywords as hashtags
        hashtags = " ".join([f"#{kw['phrase'].replace(' ', '_')}" for kw in cluster['keywords']])
        cluster_wordclouds_data.append({'cid': cid, 'label': cluster['label'], 'wc': wc, 'hashtags': hashtags})

        if not only_overview and not only_combined:
            fig, ax = plt.subplots(figsize=(4, 3.5))  # Increased height for hashtags
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            ax.set_title(f"Cluster {cid} ({cluster['label']})", fontsize=10)

            ax.text(0.5, -0.1, hashtags, ha='center', va='top', transform=ax.transAxes, fontsize=8, color='gray', wrap=True)
            plt.tight_layout(pad=0.5)
            uris[f"wordcloud-cluster-{cid}"] = _plot_to_data_uri()

    if cluster_wordclouds_data and not only_combined:
        n_clusters = len(cluster_wordclouds_data)
        cols = int(np.ceil(np.sqrt(n_clusters)))
        rows = int(np.ceil(n_clusters / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
        axes = axes.flatten()
        for i, data in enumerate(cluster_wordclouds_data):
            ax = axes[i]
            ax.imshow(data['wc'], interpolation="bilinear")
            ax.set_title(f"C{data['cid']}: {data['label']}", fontsize=10)
            t = ax.text(0.5, -0.1, data['hashtags'], ha='center', va='top', transform=ax.transAxes, fontsize=8, color='gray')
            t.set_wrap(True)
            ax.axis("off")
        for i in range(n_clusters, len(axes)):
            axes[i].axis("off")  # Hide unused subplots
        plt.suptitle("Cluster Word Cloud Overview", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.5, w_pad=1.0)
        uris["wordcloud-cluster-overview"] = _plot_to_data_uri()

    if only_combined and all_freqs:
        wc = WordCloud(width=800, height=600, background_color="white").generate_from_frequencies(all_freqs)
        plt.figure(figsize=(8, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Combined Word Cloud for All Clusters")

        # Add hashtags for the combined cloud
        all_keywords = sorted(list(set(kw['phrase'] for lc in labeled_clusters for kw in lc['keywords'])))
        combined_hashtags = " ".join([f"#{kw.replace(' ', '_')}" for kw in all_keywords[:15]])  # Limit to 15 for readability
        t = plt.text(0.5, -0.05, combined_hashtags, ha='center', va='top', transform=plt.gca().transAxes, fontsize=9, color='gray')
        t.set_wrap(True)

        uris["wordcloud-combined"] = _plot_to_data_uri()

    return uris


def make_cluster_scatter(embeddings, cluster_labels, method="pca"):
    """A 2D scatter plot of sentences, colored by cluster.

    **Details**:
    Uses either PCA (`cluster-scatter-pca`) or t-SNE (`cluster-scatter-tsne`) for dimensionality reduction.
    """
    if embeddings.shape[1] > 2:
        if method == "tsne":
            reducer = TSNE(
                n_components=2, random_state=42,  # Ensure perplexity is less than n_samples
                perplexity=min(30.0, len(embeddings) - 1),
            )
        else:
            reducer = PCA(n_components=2)
        coords = reducer.fit_transform(embeddings)
    else:
        coords = embeddings

    plt.figure(figsize=(6, 6))
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for i, label in enumerate(unique_labels):
        cluster_coords = coords[cluster_labels == label]
        plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], color=colors(i), s=50, alpha=0.7, label=f'Cluster {label}')

    for i, txt in enumerate(range(len(coords))):
        plt.text(
            coords[i, 0], coords[i, 1], f'S{i}',
            fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle='circle,pad=0.1', fc='white', ec='none', alpha=0.6)
        )

    plt.title(f"Sentence Clusters ({method.upper()})")
    plt.axis("off")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


def make_cluster_similarity(labels, n_clusters, title="Cluster Similarity"):
    """Visualizes cluster similarity based on the product of their sizes.

    **Details**:
    This is a simple metric and does not reflect semantic similarity.
    """
    similarity_matrix = np.zeros((n_clusters, n_clusters))
    unique_labels = np.unique(labels)

    for i in range(n_clusters):
        for j in range(i, n_clusters):
            sim = np.sum(labels == unique_labels[i]) * np.sum(labels == unique_labels[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(similarity_matrix, cmap='Greens')
    plt.colorbar(label='Similarity (Size Product)')
    plt.title(f'{title} (Heatmap)')
    plt.xlabel('Cluster')
    plt.ylabel('Cluster')
    plt.xticks(range(n_clusters), unique_labels)
    plt.yticks(range(n_clusters), unique_labels)

    plt.subplot(1, 2, 2)
    X, Y = np.meshgrid(range(n_clusters), range(n_clusters))
    plt.contourf(X, Y, similarity_matrix, cmap='Greens', levels=20)
    plt.colorbar(label='Similarity (Size Product)')
    plt.title(f'{title} (Contour)')
    plt.xlabel('Cluster')
    plt.ylabel('Cluster')
    plt.xticks(range(n_clusters), unique_labels)
    plt.yticks(range(n_clusters), unique_labels)

    plt.tight_layout()
    return _plot_to_data_uri()


def make_cluster_separation(embeddings, labels, n_clusters, title="Cluster Separation"):
    """A heatmap showing the average distance between all pairs of clusters.

    **Details**:
    This helps to see which clusters are semantically close and which are far apart. Lower distance values (darker colors) mean clusters are more similar.
    """
    distances = pairwise_distances(embeddings, metric='cosine')
    cluster_distances = np.zeros((n_clusters, n_clusters))
    unique_labels = np.unique(labels)

    for i in range(n_clusters):
        cluster_i_indices = np.where(labels == unique_labels[i])[0]
        for j in range(i, n_clusters):
            if i == j:
                # A cluster with one point has an intra-cluster distance of 0
                if len(cluster_i_indices) > 1:
                    cluster_distances[i, j] = np.mean(distances[np.ix_(cluster_i_indices, cluster_i_indices)])
                else:
                    cluster_distances[i, j] = 0
            else:
                cluster_j_indices = np.where(labels == unique_labels[j])[0]
                dist = np.mean(distances[np.ix_(cluster_i_indices, cluster_j_indices)])
                cluster_distances[i, j] = dist
                cluster_distances[j, i] = dist

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cluster_distances, cmap='YlOrRd')
    fig.colorbar(im, ax=ax, label='Average Cosine Distance')
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('Cluster')
    ax.set_xticks(np.arange(n_clusters))
    ax.set_yticks(np.arange(n_clusters))
    ax.set_xticklabels(unique_labels)
    ax.set_yticklabels(unique_labels)
    ax.set_xticks(np.arange(n_clusters).astype(int))
    ax.set_yticks(np.arange(n_clusters).astype(int))
    ax.set_xticklabels(unique_labels.astype(int))
    ax.set_yticklabels(unique_labels.astype(int))

    for i in range(n_clusters):
        for j in range(n_clusters):
            ax.text(j, i, f"{cluster_distances[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    return _plot_to_data_uri()


def make_cluster_graph(labeled_clusters, model):
    """A network graph showing the semantic relationship between keywords across all clusters.

    **Details**:
    Keywords are nodes, and edges connect semantically similar keywords. Clusters are represented by colored convex hulls enclosing their keywords.
    """
    if not labeled_clusters:
        return None

    all_keywords = {}  # phrase -> {clusters: set()}
    cluster_keyword_map = {}  # cluster_id -> [keywords]

    for lc in labeled_clusters:
        cid = lc['cluster_id']
        cluster_keyword_map[cid] = []
        for kw in lc['keywords']:
            phrase = kw['phrase']
            cluster_keyword_map[cid].append(phrase)
            if phrase not in all_keywords:
                all_keywords[phrase] = {'clusters': {cid}}

    unique_phrases = list(all_keywords.keys())
    if not unique_phrases:
        return None

    _, phrase_embeddings = model.encode_with_stats(unique_phrases, batch_size=64)
    phrase_embeddings = phrase_embeddings / np.linalg.norm(phrase_embeddings, axis=1, keepdims=True)

    for i, phrase in enumerate(unique_phrases):
        all_keywords[phrase]['emb'] = phrase_embeddings[i]

    G = nx.Graph()
    for phrase in unique_phrases:
        G.add_node(phrase, type='keyword')

    # Add edges based on keyword similarity
    sim_matrix = cosine_similarity(phrase_embeddings)
    threshold = 0.6  # Similarity threshold for connecting keywords
    for i in range(len(unique_phrases)):
        for j in range(i + 1, len(unique_phrases)):
            if sim_matrix[i, j] > threshold:
                G.add_edge(unique_phrases[i], unique_phrases[j], weight=sim_matrix[i, j])  # weight for layout

    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=1.5 / np.sqrt(len(G.nodes())), iterations=70, weight='weight', seed=42)

    ax = plt.gca()
    colors = plt.cm.get_cmap('tab10', len(labeled_clusters))

    for i, lc in enumerate(labeled_clusters):
        cid = lc['cluster_id']
        keywords_in_cluster = cluster_keyword_map.get(cid, [])
        if len(keywords_in_cluster) > 2:
            points = np.array([pos[kw] for kw in keywords_in_cluster if kw in pos])
            if len(points) > 2:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1], color=colors(i), linestyle='-', linewidth=1,
                                alpha=0.8)
                    ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=colors(i), alpha=0.1)
                except ImportError:
                    # Fallback if scipy is not installed.
                    logging.warning("Scipy not installed, cannot draw convex hulls for cluster graph.")
        # Add cluster label at the centroid of its keywords
        kw_positions = np.array([pos[kw] for kw in keywords_in_cluster if kw in pos])
        if len(kw_positions) > 0:
            center = kw_positions.mean(axis=0)
            ax.text(center[0], center[1], f"Cluster {cid}:\n{lc['label']}",
                    ha='center', va='center', fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc=colors(i), ec='none', alpha=0.4))

    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='black', alpha=0.7)

    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', verticalalignment='bottom')

    plt.title("Topic Keyword Relationship Graph", fontsize=16)
    plt.axis("off")
    return _plot_to_data_uri()


def make_keyword_cooccurrence_graph(labeled_clusters, title="Keyword Co-occurrence Network by Cluster"):
    """A network graph where nodes are keywords and edges connect keywords that appear together in the same cluster's keyword list.

    **Details**:
    This plot shows which keywords are associated with each other within the context of the generated topics.
    """
    G = nx.Graph()
    cluster_keyword_map = {}
    colors = plt.cm.get_cmap('tab10', len(labeled_clusters))

    for i, lc in enumerate(labeled_clusters):
        cid = lc['cluster_id']
        keywords = [kw['phrase'] for kw in lc['keywords']]
        cluster_keyword_map[cid] = keywords
        for kw in keywords:
            if not G.has_node(kw):
                G.add_node(kw, clusters={cid}, color=colors(i), size=200)
            else:
                G.nodes[kw]['clusters'].add(cid)
                G.nodes[kw]['color'] = 'grey'
                G.nodes[kw]['size'] = 400  # Shared keywords are larger

    for cid, keywords in cluster_keyword_map.items():
        from itertools import combinations
        for kw1, kw2 in combinations(keywords, 2):
            if G.has_edge(kw1, kw2):
                G[kw1][kw2]['weight'] += 1
            else:
                G.add_edge(kw1, kw2, weight=1, color='lightgray')

    if not G.nodes():
        return None

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.6, iterations=60, seed=42, weight='weight')

    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    edge_widths = [G[u][v]['weight'] * 0.8 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='lightgray', alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='black', font_weight='bold')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f"Cluster {lc['cluster_id']}: {lc['label']}",
                              markerfacecolor=colors(i), markersize=10) for i, lc in enumerate(labeled_clusters)]
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Shared Keyword',
                                  markerfacecolor='grey', markersize=10))
    plt.legend(handles=legend_elements, loc='best', title="Clusters")

    plt.title(title, fontsize=16)
    plt.axis('off')
    return _plot_to_data_uri()


def make_keyword_cooccurrence_heatmap(labeled_clusters, title="Keyword Co-occurrence Heatmap"):
    """A heatmap showing how often pairs of keywords co-occur in the same cluster's keyword list.

    **Details**:
    This is a matrix-based alternative to the co-occurrence graph.
    """
    all_keywords = sorted(list(set(kw['phrase'] for lc in labeled_clusters for kw in lc['keywords'])))
    if not all_keywords:
        return None

    keyword_to_idx = {kw: i for i, kw in enumerate(all_keywords)}
    n_keywords = len(all_keywords)
    cooccurrence_matrix = np.zeros((n_keywords, n_keywords))

    for lc in labeled_clusters:
        keywords = [kw['phrase'] for kw in lc['keywords']]
        from itertools import combinations
        for kw1, kw2 in combinations(keywords, 2):
            idx1, idx2 = keyword_to_idx[kw1], keyword_to_idx[kw2]
            cooccurrence_matrix[idx1, idx2] += 1
            cooccurrence_matrix[idx2, idx1] += 1

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cooccurrence_matrix, cmap='viridis')

    fig.colorbar(im, ax=ax, label='Co-occurrence Count')
    ax.set_title(title, fontsize=14)
    ax.set_xticks(np.arange(n_keywords))
    ax.set_yticks(np.arange(n_keywords))
    ax.set_xticklabels(all_keywords, rotation=90, fontsize=9)
    ax.set_yticklabels(all_keywords, fontsize=9)

    for i in range(n_keywords):
        for j in range(n_keywords):
            if cooccurrence_matrix[i, j] > 0:
                ax.text(j, i, int(cooccurrence_matrix[i, j]), ha="center", va="center", color="w", fontsize=7)

    plt.tight_layout()
    return _plot_to_data_uri()


def make_cluster_overlap_graph(labeled_clusters, embeddings, labels, title="Cluster Similarity Network"):
    """A network graph where nodes are clusters and edge weights represent the similarity between cluster centroids.

    **Details**:
    This plot provides a high-level overview of how the identified topics relate to each other.
    """
    n_clusters = len(labeled_clusters)
    unique_labels = np.unique(labels)

    centroids = np.array([embeddings[labels == label].mean(axis=0) for label in unique_labels])
    sim_matrix = cosine_similarity(centroids)

    G = nx.Graph()
    colors = plt.cm.get_cmap('tab10', n_clusters)

    for i, lc in enumerate(labeled_clusters):
        G.add_node(lc['cluster_id'], label=lc['label'], size=lc['size'], color=colors(i))

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            similarity = sim_matrix[i, j]
            if similarity > 0.3:  # Threshold to avoid a fully connected graph
                G.add_edge(unique_labels[i], unique_labels[j], weight=similarity)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.8, weight='weight', seed=42)

    node_sizes = [G.nodes[n]['size'] * 100 for n in G.nodes()]
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    node_labels = {n: f"{n}\n({G.nodes[n]['label']})" for n in G.nodes()}

    nx.draw(G, pos, with_labels=False, node_size=node_sizes, node_color=node_colors, width=edge_weights, edge_color='gray', alpha=0.8)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black', verticalalignment='center')
    plt.title(title, fontsize=16)
    return _plot_to_data_uri()


def make_bipartite_cluster_sentence_graph(labeled_clusters, embeddings, labels, title="Bipartite Cluster-Sentence Graph"):
    """A two-part graph showing cluster nodes on one side and sentence nodes on the other, with lines indicating membership.

    **Details**:
    Also draws edges between similar sentences from different clusters to show overlap.
    """
    if not labeled_clusters:
        return None

    G = nx.Graph()
    n_clusters = len(labeled_clusters)
    n_sentences = len(embeddings)
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', n_clusters)

    cluster_nodes = [f"C{lc['cluster_id']}" for lc in labeled_clusters]
    sentence_nodes = [f"S{i}" for i in range(n_sentences)]

    for i, lc in enumerate(labeled_clusters):
        node_id = f"C{lc['cluster_id']}"
        G.add_node(node_id, type='cluster', bipartite=0, color=colors(i), size=lc['size'] * 100, label=f"{node_id}: {lc['label']}")

    for i in range(n_sentences):
        cluster_id = labels[i]
        cluster_color_index = np.where(unique_labels == cluster_id)[0][0]
        G.add_node(f"S{i}", type='sentence', bipartite=1, color=colors(cluster_color_index), size=50)

    for i in range(n_sentences):
        G.add_edge(f"S{i}", f"C{labels[i]}", type='membership')

    sim_matrix = cosine_similarity(embeddings)
    overlap_threshold = 0.75  # Similarity threshold for drawing an overlap edge
    for i in range(n_sentences):
        for j in range(i + 1, n_sentences):
            if labels[i] != labels[j] and sim_matrix[i, j] > overlap_threshold:
                G.add_edge(f"S{i}", f"S{j}", type='overlap', weight=sim_matrix[i, j])

    plt.figure(figsize=(18, 14))
    pos = {}
    pos.update((node, (1, i)) for i, node in enumerate(cluster_nodes))
    pos.update((node, (2, i)) for i, node in enumerate(sentence_nodes))

    pos = nx.spring_layout(G, pos=pos, fixed=cluster_nodes, iterations=50, k=0.3)

    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)

    membership_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'membership']
    overlap_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'overlap']
    nx.draw_networkx_edges(G, pos, edgelist=membership_edges, edge_color='lightgray', alpha=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=overlap_edges, edge_color='red', width=1.5, style='dashed', alpha=0.7)

    cluster_labels = {n: G.nodes[n]['label'] for n in G.nodes if G.nodes[n]['type'] == 'cluster'}
    sentence_labels = {n: n for n in G.nodes if G.nodes[n]['type'] == 'sentence'}
    nx.draw_networkx_labels(G, pos, labels=cluster_labels, font_size=12, font_weight='bold')
    nx.draw_networkx_labels(G, pos, labels=sentence_labels, font_size=8)

    plt.title(title, fontsize=18)
    plt.axis('off')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Sentence Overlap (>0.75 similarity)'),
        Line2D([0], [0], color='gray', lw=2, label='Cluster Membership')
    ]
    plt.legend(handles=legend_elements, loc='best')

    return _plot_to_data_uri()


def make_treemap_plot(clustering_model, labeled_clusters, title="Hierarchical Cluster Treemap"):
    """A treemap where each rectangle's area is proportional to the size of a cluster.

    **Details**:
    This visualization is effective for comparing the relative sizes of many clusters at once. (Requires `agglomerative` algorithm).
    """
    if not hasattr(clustering_model, 'children_'):
        return None  # Not a hierarchical model

    try:
        import squarify
    except ImportError:
        return None  # squarify not installed

    sizes = [lc['size'] for lc in labeled_clusters]
    labels = [f"C{lc['cluster_id']}\n{lc['label']}\n(n={lc['size']})" for lc in labeled_clusters]
    colors = plt.cm.get_cmap('tab20c', len(labeled_clusters))

    plt.figure(figsize=(12, 8))
    squarify.plot(sizes=sizes, label=labels, color=[colors(i) for i in range(len(labels))],
                  alpha=.8, text_kwargs={'fontsize': 10, 'color': 'black'})
    plt.title(title, fontsize=16)
    plt.axis('off')

    return _plot_to_data_uri()


def make_dendrogram_heatmap(clustering_model, embeddings, title="Sentence Similarity Dendrogram & Heatmap"):
    """A heatmap of the sentence similarity matrix, accompanied by a dendrogram showing the hierarchical merging of sentences.

    **Details**:
    This is a detailed, low-level view of how the `agglomerative` algorithm builds the clusters. (Requires `agglomerative` algorithm).
    """
    if not hasattr(clustering_model, 'children_'):
        return None

    n_samples = len(embeddings)
    counts = np.zeros(clustering_model.children_.shape[0])
    for i, merge in enumerate(clustering_model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([clustering_model.children_, clustering_model.distances_, counts]).astype(float)

    similarity_matrix = cosine_similarity(embeddings)

    try:
        import seaborn as sns
        g = sns.clustermap(similarity_matrix,
                           row_linkage=linkage_matrix,
                           col_linkage=linkage_matrix,
                           cmap='viridis',
                           figsize=(10, 10))
        # Add sentence labels to the heatmap axes, ensuring ticks and labels match.
        reordered_indices = g.dendrogram_row.reordered_ind  # The reordered_ind gives the order of the original samples.
        g.ax_heatmap.set_xticks(np.arange(len(reordered_indices)))
        g.ax_heatmap.set_yticks(np.arange(len(reordered_indices)))
        g.ax_heatmap.set_xticklabels([f'S{i}' for i in reordered_indices], rotation=90, fontsize=8)
        g.ax_heatmap.set_yticklabels([f'S{i}' for i in reordered_indices], rotation=0, fontsize=8)
        g.ax_heatmap.set_xlabel("Sentences (reordered by similarity)")
        g.ax_heatmap.set_ylabel("Sentences (reordered by similarity)")

        g.fig.suptitle(title, fontsize=16, y=1.02)

        g.ax_col_dendrogram.set_title("Hierarchical Clustering (Dendrogram)", fontsize=10, pad=10)
        g.fig.text(0.5, 0.92,
                   "The tree shows how sentences are grouped. Shorter branches mean higher similarity.",
                   ha='center', va='center', fontsize=9, style='italic', color='gray')

        cbar = g.ax_heatmap.collections[0].colorbar
        cbar.set_label("Sentence Similarity (Cosine)", rotation=270, labelpad=15)
        plt.tight_layout()
        g.fig.text(0.5, 0.02,
                   "The heatmap shows sentence-pair similarity. Brighter colors mean more similar.",
                   ha='center', va='center', fontsize=9, style='italic', color='gray')

        return _plot_to_data_uri()
    except ImportError:
        return None
    plt.close()


def make_cluster_distance_distribution_plot(embeddings, labels, title="Cluster Distance Distributions"):
    """Box plots comparing the distribution of distances *within* each cluster (intra-cluster) to the distances *between* clusters (inter-cluster).

    **Details**:
    Well-separated clusters should have low intra-cluster distances and high inter-cluster distances.
    """
    unique_labels = sorted(np.unique(labels))
    n_clusters = len(unique_labels)
    distances = pairwise_distances(embeddings, metric='cosine')

    plot_data = []
    plot_labels = []
    colors = []
    palette = plt.cm.get_cmap('tab10', n_clusters)

    num_comparisons = 0
    for i, label1 in enumerate(unique_labels):
        indices1 = np.where(labels == label1)[0]
        if len(indices1) > 1:
            intra_distances = distances[np.ix_(indices1, indices1)][np.triu_indices(len(indices1), k=1)]
            plot_data.append(intra_distances)
            plot_labels.append(f"C{label1}\n(Intra)")
            colors.append(palette(i))  # Use cluster-specific color
            num_comparisons += 1

        for j in range(i + 1, n_clusters):
            label2 = unique_labels[j]
            indices2 = np.where(labels == label2)[0]
            inter_distances = distances[np.ix_(indices1, indices2)].ravel()
            plot_data.append(inter_distances)
            plot_labels.append(f"C{label1} vs C{label2}\n(Inter)")
            colors.append('lightgrey')  # Use a neutral color for inter-cluster
            num_comparisons += 1

    try:
        import seaborn as sns
        fig_height = max(8, num_comparisons * 0.4)
        plt.figure(figsize=(12, fig_height))

        ax = sns.boxplot(data=plot_data, orient='h', showfliers=False, palette=colors)
        sns.stripplot(data=plot_data, orient='h', color=".25", size=2, alpha=0.3)
        ax.set_yticks(ax.get_yticks(), plot_labels)

        for i, d in enumerate(plot_data):
            if len(d) > 0:
                median_val = np.median(d)
                ax.text(median_val, i, f'{median_val:.2f}',
                        verticalalignment='center', size='small', color='black', weight='semibold',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

        plt.title(title, fontsize=14)
        plt.xlabel("Cosine Distance")
        plt.tight_layout()
        return _plot_to_data_uri()
    except ImportError:
        return None
    plt.close()


def make_keyword_contribution_plot(labeled_clusters, vectorizer: TfidfVectorizer, tfidf_matrix, title="Keyword Contribution & Exclusivity"):
    """A scatter plot showing each keyword's contribution (mean TF-IDF) vs. its exclusivity to its cluster.

    **Details**:
    This helps identify keywords that are both important within a cluster and unique to it.
    """
    if not labeled_clusters:
        return None

    try:
        vocab = vectorizer.get_feature_names_out()
        keyword_to_idx = {kw: i for i, kw in enumerate(vocab)}
    except ValueError:
        return None  # Corpus might be empty or too small

    plot_data = []
    for lc in labeled_clusters:
        cid = lc['cluster_id']
        cluster_indices = lc['sentence_ids']
        cluster_docs_tfidf = tfidf_matrix[cluster_indices]

        for kw_data in lc['keywords']:
            kw = kw_data['phrase']
            if kw in keyword_to_idx:
                kw_idx = keyword_to_idx[kw]

                contribution = cluster_docs_tfidf[:, kw_idx].mean()

                total_contribution = tfidf_matrix[:, kw_idx].mean()
                exclusivity = contribution / (total_contribution + 1e-9)

                plot_data.append({
                    'keyword': kw,
                    'cluster': f"C{cid}",
                    'contribution': contribution,
                    'exclusivity': exclusivity
                })

    if not plot_data:
        return None

    df = pd.DataFrame(plot_data)

    try:
        import seaborn as sns
        plt.figure(figsize=(12, 8))
        ax = sns.scatterplot(data=df, x='exclusivity', y='contribution', hue='cluster', size='contribution',
                             sizes=(50, 500), alpha=0.7, style='cluster')

        for i, row in df.iterrows():
            ax.text(row['exclusivity'] + 0.02, row['contribution'], row['keyword'], fontsize=8)

        plt.title(title)
        plt.xlabel("Exclusivity (Higher is more unique to cluster)")
        plt.ylabel("Contribution (Mean TF-IDF within cluster)")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        return _plot_to_data_uri()
    except ImportError:
        return None
    plt.close()


def make_characteristic_keywords_plot(labeled_clusters, clusters, corpus, vectorizer: TfidfVectorizer, tfidf_matrix, title="Characteristic Keywords per Cluster"):
    """Bar charts showing the most uniquely characteristic keywords for each cluster.

    **Details**:
    A characteristic keyword is one that is much more common inside the cluster than outside of it.
    """
    if not labeled_clusters:
        return None

    try:
        vocab = vectorizer.get_feature_names_out()
        keyword_to_idx = {kw: i for i, kw in enumerate(vocab)}
    except ValueError:
        return None  # Corpus might be empty or too small

    # This score is high if a keyword is frequent in its cluster but rare elsewhere.
    plot_data = []
    for lc in labeled_clusters:
        cluster_label = f"C{lc['cluster_id']}: {lc['label']}"
        cid = lc['cluster_id']
        cluster_indices = clusters[cid]['indices']
        other_indices = [i for i, _ in enumerate(corpus) if i not in cluster_indices]

        in_cluster_tfidf = np.asarray(tfidf_matrix[cluster_indices].mean(axis=0)).ravel()
        out_cluster_tfidf = np.asarray(tfidf_matrix[other_indices].mean(axis=0)).ravel()

        characteristic_score = in_cluster_tfidf - out_cluster_tfidf

        # Get top 5 characteristic keywords for this cluster
        top_indices = np.argsort(characteristic_score)[-5:]
        for idx in top_indices:
            plot_data.append({
                'keyword': vocab[idx],
                'cluster': cluster_label,
                'score': characteristic_score[idx]
            })

    if not plot_data:
        return None

    df = pd.DataFrame(plot_data)
    df = df.sort_values(by=['cluster', 'score'], ascending=[True, True])

    try:
        import seaborn as sns

        unique_clusters = df['cluster'].unique()
        palette = {cluster: plt.cm.tab10(i % 10) for i, cluster in enumerate(unique_clusters)}

        g = sns.FacetGrid(df, col="cluster", col_wrap=min(3, len(df['cluster'].unique())),
                          sharex=False, sharey=False, height=4, aspect=1.2)

        g.map_dataframe(sns.barplot, x="score", y="keyword", orient='h', hue="cluster", palette=palette, legend=False)

        g.fig.suptitle(title, y=1.03)
        g.set_axis_labels("Characteristic Score (Higher is more unique)", "Keyword")
        g.set_titles("Cluster: {col_name}")

        return _plot_to_data_uri()
    except ImportError:
        return None
    plt.close()


def make_word_cooccurrence_graph(labeled_clusters, candidate_meta_by_cluster, title="Word Co-occurrence Network (All Candidates)", max_nodes=75):
    """A network graph of all *candidate* words (not just the final keywords), showing co-occurrence within clusters.

    **Details**:
    Nodes are words, and edges represent co-occurrence within the same cluster's candidate list.
    Shared words are highlighted, and node size is proportional to its connectivity.
    """
    G = nx.Graph()
    cluster_word_map = {}  # cid -> [words]

    palette = plt.cm.get_cmap('tab20', max(20, len(labeled_clusters)))
    cluster_id_to_color = {lc['cluster_id']: palette(i % 20) for i, lc in enumerate(labeled_clusters)}

    for lc in labeled_clusters:
        cid = lc['cluster_id']
        if cid in candidate_meta_by_cluster:
            _, _, _, all_phrases = candidate_meta_by_cluster[cid]
            cluster_word_map[cid] = all_phrases
            for word in all_phrases:
                if not G.has_node(word):
                    G.add_node(word, clusters={cid}, color=cluster_id_to_color[cid])
                else:
                    G.nodes[word]['clusters'].add(cid)
                    G.nodes[word]['color'] = 'grey'  # Shared word

    for cid, words in cluster_word_map.items():
        from itertools import combinations
        for w1, w2 in combinations(words, 2):
            if G.has_edge(w1, w2):
                G[w1][w2]['weight'] += 1  # This edge represents co-occurrence in multiple clusters; mark it as shared.
                G[w1][w2]['color'] = 'black'  # Shared co-occurrence edge
            else:
                G.add_edge(w1, w2, weight=1, color=cluster_id_to_color[cid])

    # Filter graph if it's too large for readability.
    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.degree, key=lambda item: item[1], reverse=True)[:max_nodes]
        top_node_names = [node for node, degree in top_nodes]
        G = G.subgraph(top_node_names)

    if not G.nodes():
        return None

    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.9 / np.sqrt(G.number_of_nodes()), iterations=80, seed=42, weight='weight')

    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    # Node size proportional to its degree (connectivity).
    degrees = [G.degree(n) for n in G.nodes()]
    min_size, max_size = 100, 800
    if degrees and max(degrees) > min(degrees):
        node_sizes = [min_size + (d - min(degrees)) * (max_size - min_size) / (max(degrees) - min(degrees)) for d in degrees]
    else:
        node_sizes = [min_size] * len(degrees)

    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_widths = [G[u][v]['weight'] * 0.6 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)

    # Use adjustText to prevent label overlap
    try:
        from adjustText import adjust_text
        texts = [plt.text(pos[n][0], pos[n][1], n, fontsize=8, ha='center', va='center') for n in G.nodes()]
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))
    except ImportError:
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f"Cluster {lc['cluster_id']}: {lc['label']}",
                              markerfacecolor=cluster_id_to_color[lc['cluster_id']], markersize=10) for i, lc in enumerate(labeled_clusters)]
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Shared Word',
                                  markerfacecolor='grey', markersize=10))
    legend_elements.append(Line2D([0], [0], color='black', lw=2, label='Shared Co-occurrence'))

    plt.legend(handles=legend_elements, loc='best', title="Clusters")

    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    return _plot_to_data_uri()


def make_word_cooccurrence_heatmap(labeled_clusters, candidate_meta_by_cluster, title="Word Co-occurrence Heatmap (All Candidates)", max_words=50):
    """A heatmap of all *candidate* words, showing co-occurrence frequency.

    **Details**:
    Cells are annotated with the count and the IDs of clusters where the co-occurrence happens.
    """
    all_words = set()
    for cid in candidate_meta_by_cluster:
        _, _, _, all_phrases = candidate_meta_by_cluster[cid]
        all_words.update(all_phrases)

    if not all_words:
        return None

    # Build a temporary graph to find the most connected words for readability.
    temp_G = nx.Graph()
    cooccurrence_context = {}  # (w1, w2) -> {cluster_ids}
    for cid in candidate_meta_by_cluster:
        _, _, _, all_phrases = candidate_meta_by_cluster[cid]
        from itertools import combinations
        for w1, w2 in combinations(all_phrases, 2):
            key = tuple(sorted((w1, w2)))
            cooccurrence_context.setdefault(key, set()).add(cid)
            temp_G.add_edge(w1, w2)

    if temp_G.number_of_nodes() > max_words and max_words > 0:
        top_nodes = sorted(temp_G.degree, key=lambda item: item[1], reverse=True)[:max_words]
        sorted_words = sorted([node for node, degree in top_nodes])
    else:
        sorted_words = sorted(list(all_words))

    if not sorted_words:
        return None  # No words to plot.

    word_to_idx = {word: i for i, word in enumerate(sorted_words)}
    n_words = len(sorted_words)
    cooccurrence_matrix = np.zeros((n_words, n_words), dtype=int)
    annotation_matrix = [['' for _ in range(n_words)] for _ in range(n_words)]

    for (w1, w2), cluster_ids in cooccurrence_context.items():
        if w1 in word_to_idx and w2 in word_to_idx:
            idx1, idx2 = word_to_idx[w1], word_to_idx[w2]
            count = len(cluster_ids)
            cooccurrence_matrix[idx1, idx2] = count
            cooccurrence_matrix[idx2, idx1] = count

            # Annotation text with count and cluster IDs
            cluster_str = ",".join(map(str, sorted(list(cluster_ids))))
            annotation = f"{count}\n(C:{cluster_str})"
            annotation_matrix[idx1][idx2] = annotation
            annotation_matrix[idx2][idx1] = annotation

    # Increase the size factor to make boxes larger and labels more readable.
    fig, ax = plt.subplots(figsize=(max(12, n_words * 0.45), max(10, n_words * 0.45)))
    # Use a logarithmic color scale to handle skewed data, adding a small constant to avoid log(0)
    from matplotlib.colors import LogNorm
    im = ax.imshow(cooccurrence_matrix + 1e-9, cmap='viridis', norm=LogNorm())

    fig.colorbar(im, ax=ax, label='Co-occurrence Count')
    ax.set_title(f"{title} (Top {n_words} Words)", fontsize=14)
    ax.set_xticks(np.arange(n_words))
    ax.set_yticks(np.arange(n_words))
    ax.set_xticklabels(sorted_words, rotation=90, fontsize=9)
    ax.set_yticklabels(sorted_words, fontsize=9)

    threshold = im.norm(cooccurrence_matrix.max()) / 2.
    text_colors = ["black", "white"]
    for i in range(n_words):
        for j in range(n_words):
            if cooccurrence_matrix[i, j] > 0:
                color = text_colors[int(im.norm(cooccurrence_matrix[i, j]) > threshold)]
                ax.text(j, i, annotation_matrix[i][j], ha="center", va="center", color=color, fontsize=6)

    plt.tight_layout()
    return _plot_to_data_uri()


def _wrap_text(text, width):
    """A helper function to wrap text for node labels."""
    import textwrap
    return '\n'.join(textwrap.wrap(text, width=width))


def _hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    From https://stackoverflow.com/a/29597209/29597209
    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)
    root: the root node of current branch
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is not directed and this is not given, a random node will be used
    width: horizontal space allocated for this branch - avoids overlap with other branches
    vert_gap: gap between levels of hierarchy
    vert_loc: vertical location of root
    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        return nx.spring_layout(G)  # Fallback for non-tree graphs
    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = list(G.nodes)[0]

    def _hierarchy_pos_recursive(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if parent is not None and parent in children:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos_recursive(G, child, width=dx, vert_gap=vert_gap,
                                               vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                               pos=pos, parent=root)
        return pos

    return _hierarchy_pos_recursive(G, root, width=width, vert_gap=vert_gap, vert_loc=vert_loc, xcenter=xcenter)


def make_label_word_semantic_hierarchy(labeled_clusters, candidate_meta_by_cluster, title="Label-Word Semantic Hierarchy"):
    """A hierarchical tree diagram showing the relationship from the root, to cluster labels, to their top keywords.

    **Details**:
    This plot helps visualize the topic taxonomy that has been discovered.
    """
    if not labeled_clusters:
        return None

    G = nx.DiGraph()
    root_node = "Root"
    G.add_node(root_node, level=0, label=root_node, size=3000)

    cluster_embeddings = {}
    word_embeddings = {}
    for lc in labeled_clusters:
        cid = lc['cluster_id']
        if cid in candidate_meta_by_cluster:
            candidates, cand_emb, _, _ = candidate_meta_by_cluster[cid]
            cluster_emb = cand_emb.mean(axis=0)
            cluster_embeddings[lc['label']] = cluster_emb / np.linalg.norm(cluster_emb)
            for i, word in enumerate(candidates):
                word_embeddings[word] = cand_emb[i]

    # Sort clusters by their similarity to the global mean (less specific to more specific)
    global_mean = np.mean(list(cluster_embeddings.values()), axis=0)
    sorted_labels = sorted(cluster_embeddings.keys(), key=lambda label: cosine_similarity(cluster_embeddings[label].reshape(1, -1), global_mean.reshape(1, -1))[0][0])

    palette = plt.cm.get_cmap('tab20', len(labeled_clusters))
    cluster_label_to_color = {label: palette(i % 20) for i, label in enumerate(sorted_labels)}

    for label in sorted_labels:
        cid = next(lc['cluster_id'] for lc in labeled_clusters if lc['label'] == label)
        cluster_size = next(lc['size'] for lc in labeled_clusters if lc['cluster_id'] == cid)
        node_size = 1500 + cluster_size * 50  # Size proportional to cluster size
        wrapped_label = _wrap_text(f"C{cid}: {label}", 15)
        G.add_node(label, level=1, label=wrapped_label, size=node_size, color=cluster_label_to_color[label])
        G.add_edge(root_node, label)

        # Find top N words for this cluster to display
        candidates, cand_emb, scores, _ = candidate_meta_by_cluster[cid]
        top_indices = np.argsort(scores)[-5:]  # Top 5 words

        for i in top_indices:
            word = candidates[i]
            wrapped_word = _wrap_text(word, 12)
            # Ensure keyword nodes are unique if they appear in multiple clusters
            node_id = f"{word}_{cid}"
            G.add_node(node_id, level=2, label=wrapped_word, size=800, color=cluster_label_to_color[label])
            G.add_edge(label, node_id)

    plt.figure(figsize=(18, 12))

    pos = _hierarchy_pos(G, root=root_node, vert_gap=0.15, width=2)

    level_colors = {0: '#6c757d', 1: 'cluster_color', 2: '#e9ecef'}
    node_colors = [level_colors[0] if G.nodes[n]['level'] == 0 else G.nodes[n]['color'] for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    node_labels = {n: G.nodes[n]['label'] for n in G.nodes()}

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, node_shape='o', alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrowstyle='-', alpha=0.3, edge_color='gray', node_size=node_sizes, connectionstyle='arc3,rad=0.1')

    # Draw labels with better styling and prevent overlap
    try:
        from adjustText import adjust_text
        texts = []
        for node, (x, y) in pos.items():
            level = G.nodes[node]['level']
            font_color = 'white' if level < 2 else 'black'
            font_weight = 'bold' if level < 2 else 'normal'
            font_size = 9 if level == 1 else (10 if level == 0 else 8)
            texts.append(plt.text(x, y, node_labels[node], ha='center', va='center', color=font_color, fontsize=font_size, weight=font_weight))
        adjust_text(texts)
    except ImportError:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color='black')

    plt.title(title, fontsize=16)
    plt.axis('off')
    return _plot_to_data_uri()


def make_cluster_cohesion_plot(labeled_clusters, title="Cluster Cohesion (Size vs. Spread)"):
    """A bubble chart plotting cluster size vs. spread (average internal distance), where bubble size represents the cluster's radius.

    **Details**:
    This plot helps to quickly identify large, tight clusters (bottom-right) versus small, sparse clusters (top-left).
    """
    if not labeled_clusters:
        return None

    sizes = [lc['size'] for lc in labeled_clusters]
    avg_dists = [lc['avg_dist'] for lc in labeled_clusters]
    labels = [f"C{lc['cluster_id']}: {lc['label']}" for lc in labeled_clusters]
    radii = [lc['radius'] for lc in labeled_clusters]

    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    scatter = ax.scatter(sizes, avg_dists, s=[r * 1000 + 50 for r in radii], alpha=0.7,
                         c=range(len(labeled_clusters)), cmap='tab10')

    for i, label in enumerate(labels):
        ax.text(sizes[i], avg_dists[i], label, fontsize=9, ha='center', va='bottom')

    plt.title(title, fontsize=16)
    plt.xlabel("Cluster Size (Number of Sentences)", fontsize=12)
    plt.ylabel("Average Intra-Cluster Distance (Spread)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Create a legend for bubble size (radius)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Bubble size ~ Cluster Radius (Max Distance)',
                              markerfacecolor='gray', markersize=10)]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    return _plot_to_data_uri()


def make_cluster_outlier_scores(labeled_clusters, embeddings, labels, title="Sentence Outlier Scores within Clusters"):
    """Bar charts showing the sentences in each cluster that are least similar to their cluster's center (i.e., the biggest outliers).

    **Details**:
    This is useful for quality checking, as outliers may indicate poorly formed clusters or sentences that don't belong.
    """
    if not labeled_clusters:
        return None

    outlier_data = []
    for lc in labeled_clusters:
        cid = lc['cluster_id']
        sims = lc.get('sentences_centroid_similarity')
        if sims:
            for i, sent_id in enumerate(lc['sentence_ids']):
                # Outlier score is 1 - similarity to centroid
                outlier_score = 1 - sims[i]
                outlier_data.append({'sentence_id': f"S{sent_id}", 'cluster_id': f"C{cid}", 'score': outlier_score})

    if not outlier_data:
        return None

    df = pd.DataFrame(outlier_data)
    df = df.sort_values(by=['cluster_id', 'score'], ascending=[True, False])

    try:
        import seaborn as sns
        g = sns.FacetGrid(df, col="cluster_id", col_wrap=min(4, len(df['cluster_id'].unique())),
                          sharex=False, sharey=False, height=4)
        g.map_dataframe(sns.barplot, x="score", y="sentence_id", orient='h', color='skyblue')
        g.set_axis_labels("Outlier Score (1 - Similarity to Centroid)", "Sentence ID")
        g.set_titles("Cluster: {col_name}")
        g.fig.suptitle(title, y=1.03)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return _plot_to_data_uri()
    except ImportError:
        return None


def make_sentence_similarity_distribution(embeddings, title="Distribution of Pairwise Sentence Similarities"):
    """A histogram showing the distribution of similarity scores between all pairs of sentences in the corpus.

    **Details**:
    This gives a sense of the overall semantic diversity of the input text.
    """
    n_sentences = embeddings.shape[0]
    batch_size = 512  # Process in chunks to avoid large memory allocation
    similarities = []

    for i in range(0, n_sentences, batch_size):
        batch_embeddings = embeddings[i:i + batch_size]
        sim_batch = cosine_similarity(batch_embeddings, embeddings)

        # Extract upper triangle for each row in the batch
        for j in range(sim_batch.shape[0]):
            row_idx = i + j
            # We only need similarities for pairs (row_idx, k) where k > row_idx
            similarities.extend(sim_batch[j, row_idx + 1:])

    similarities = np.array(similarities)

    plt.figure(figsize=(8, 5))
    plt.hist(similarities, bins=50, color='steelblue', alpha=0.8)
    plt.title(title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    if len(similarities) > 0:
        plt.axvline(np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.2f}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return _plot_to_data_uri()


def make_cluster_label_similarity_heatmap(labeled_clusters, keywords_options, title="Cluster-Label Similarity Heatmap"):
    """If `predefined_labels` are provided, this heatmap shows the similarity of each cluster to each predefined label.

    **Details**:
    This is useful for evaluating how well the discovered clusters align with a known set of topics.
    """
    predefined_labels = keywords_options.get('labels')
    if not predefined_labels or not any(predefined_labels) or not labeled_clusters:
        return None

    model, _ = models.get_tracked(VectorTextModel.id, InferTracker())
    result = model.encode_with_stats(predefined_labels)
    if result is None: return None
    _, predefined_emb = result
    predefined_emb /= np.linalg.norm(predefined_emb, axis=1, keepdims=True)

    cluster_names = []
    cluster_centroids = []

    for lc in labeled_clusters:
        # Re-calculating centroid in batches to be memory-safe
        cluster_sentences = lc['sentences']
        if not cluster_sentences: continue
        batch_size = 128
        temp_res = model.encode_with_stats([cluster_sentences[0]])
        if not temp_res or temp_res[1] is None: continue  # Skip cluster if first sentence can't be encoded
        temp_emb, _ = temp_res
        centroid_sum = np.zeros(temp_emb.shape[1] if temp_emb is not None else 0)
        for i in range(0, len(cluster_sentences), batch_size):
            batch = cluster_sentences[i:i + batch_size]
            _, batch_embeddings = model.encode_with_stats(batch, normalize=True)
            centroid_sum += batch_embeddings.sum(axis=0) if batch_embeddings is not None and batch_embeddings.size > 0 else 0

        centroid = (centroid_sum / len(cluster_sentences)).reshape(1, -1)
        cluster_names.append(f"C{lc['cluster_id']}: {lc['label']}")
        cluster_centroids.append(centroid.ravel())

    if not cluster_centroids:
        return None

    sim_matrix = cosine_similarity(np.array(cluster_centroids), predefined_emb)

    fig, ax = plt.subplots(figsize=(max(8, len(predefined_labels)), max(6, len(labeled_clusters) * 0.5)))
    im = ax.imshow(sim_matrix, cmap='viridis')

    ax.set_xticks(np.arange(len(predefined_labels)))
    ax.set_yticks(np.arange(len(cluster_names)))
    ax.set_xticklabels(predefined_labels, rotation=45, ha="right")
    ax.set_yticklabels(cluster_names)

    for i in range(len(cluster_names)):
        for j in range(len(predefined_labels)):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", color="w")

    fig.colorbar(im, label="Cosine Similarity")
    plt.title(title)
    plt.tight_layout()
    return _plot_to_data_uri()


def make_keyword_rarity_plot(labeled_clusters, candidate_meta_by_cluster, corpus, title="Keyword Rarity: Cluster vs. Global Frequency"):
    """A scatter plot comparing a keyword's frequency within its cluster to its frequency in the entire corpus.

    **Details**:
    Keywords that are frequent in their cluster but rare globally (top-left quadrant) are often the most descriptive and interesting.
    """
    if not labeled_clusters or not candidate_meta_by_cluster:
        return None

    # Join corpus and cluster sentences into single lowercase strings for efficient counting.
    global_text = " ".join(corpus).lower()

    plot_data = []
    for lc in labeled_clusters:
        cid = lc['cluster_id']
        cluster_text = " ".join(lc['sentences']).lower()

        for kw in lc['keywords']:
            phrase_lower = kw['phrase'].lower()

            # Accurately count the occurrences of the full phrase.
            cluster_freq = cluster_text.count(phrase_lower)
            global_freq = global_text.count(phrase_lower)

            if cluster_freq > 0:
                # Rarity is high if the keyword appears often in the cluster but rarely elsewhere.
                rarity_score = cluster_freq / (global_freq + 1e-9)
                plot_data.append({
                    'keyword': kw['phrase'],
                    'cluster': f"C{cid}: {lc['label']}",
                    # Add 1 for log scaling to avoid log(0)
                    'cluster_freq': cluster_freq + 1,
                    'global_freq': global_freq + 1,
                    'rarity': rarity_score
                })

    if not plot_data:
        return None

    df = pd.DataFrame(plot_data)

    try:
        import seaborn as sns
        from adjustText import adjust_text

        plt.figure(figsize=(12, 8))
        ax = sns.scatterplot(
            data=df,
            x='global_freq',
            y='cluster_freq',
            hue='cluster',
            size='rarity',
            sizes=(50, 800),  # Increase max size for more impact
            alpha=0.7,
            palette='tab10'
        )

        # Use adjustText to prevent label overlap
        texts = [ax.text(row['global_freq'], row['cluster_freq'], row['keyword'], fontsize=9)
                 for _, row in df.iterrows()]
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

        plt.title(title, fontsize=14)
        plt.xlabel("Keyword Frequency in Full Corpus (Log Scale)")
        plt.ylabel("Keyword Frequency in Cluster (Log Scale)")
        plt.xscale('log', nonpositive='clip')
        plt.yscale('log', nonpositive='clip')
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        return _plot_to_data_uri()
    except ImportError:
        return None


def make_cluster_keyword_bipartite_graph(labeled_clusters, title="Cluster-Keyword Bipartite Graph"):
    """A two-part graph connecting clusters to their final keywords.

    **Details**:
    This provides a clear, direct mapping of which keywords were assigned to which clusters.
    """
    if not labeled_clusters:
        return None

    G = nx.Graph()
    cluster_nodes = []
    keyword_nodes = set()

    for lc in labeled_clusters:
        c_node = f"C{lc['cluster_id']}: {lc['label']}"
        cluster_nodes.append(c_node)
        G.add_node(c_node, type='cluster', size=lc['size'] * 50)
        for kw in lc['keywords']:
            kw_node = kw['phrase']
            keyword_nodes.add(kw_node)
            G.add_node(kw_node, type='keyword')
            G.add_edge(c_node, kw_node, weight=kw['score'])

    plt.figure(figsize=(14, 10))
    pos = nx.bipartite_layout(G, cluster_nodes)

    cluster_sizes = [G.nodes[n]['size'] for n in cluster_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes, node_color='skyblue', node_size=cluster_sizes)
    nx.draw_networkx_nodes(G, pos, nodelist=list(keyword_nodes), node_color='lightgreen', node_size=800)

    edge_widths = [d['weight'] * 5 for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6)

    nx.draw_networkx_labels(G, pos, font_size=9)

    plt.title(title, fontsize=16)
    plt.axis('off')
    return _plot_to_data_uri()


_plotter_fn = {
    'cluster-scatter-pca': make_cluster_scatter,  # todo: duplicates desc. in docs
    'cluster-scatter-tsne': make_cluster_scatter,  # todo: duplicates desc. in docs
    'cluster-density': make_cluster_density,
    'cluster-heatmap': make_cluster_heatmap,
    'cluster-sizes': make_cluster_size_bar,
    'embedding-histogram': make_embedding_histogram,
    'silhouette': make_silhouette_plot,
    'wordcloud': make_topic_wordclouds,  # todo: duplicates desc. in docs
    'wordcloud-overview': make_topic_wordclouds,  # todo: duplicates desc. in docs
    'wordcloud-combined': make_topic_wordclouds,  # todo: duplicates desc. in docs
    'graph': make_cluster_graph,
    'cluster-similarity': make_cluster_similarity,
    'cluster-separation': make_cluster_separation,
    'keyword-cooccurrence-graph': make_keyword_cooccurrence_graph,
    'keyword-cooccurrence-heatmap': make_keyword_cooccurrence_heatmap,
    'cluster-overlap-graph': make_cluster_overlap_graph,
    'bipartite-cluster-sentence-graph': make_bipartite_cluster_sentence_graph,
    'treemap': make_treemap_plot,
    'dendrogram-heatmap': make_dendrogram_heatmap,
    'cluster-distance-distribution': make_cluster_distance_distribution_plot,
    'keyword-contribution': make_keyword_contribution_plot,
    'characteristic-keywords': make_characteristic_keywords_plot,
    'word-cooccurrence-graph': make_word_cooccurrence_graph,
    'word-cooccurrence-heatmap': make_word_cooccurrence_heatmap,
    'label-word-semantic-hierarchy': make_label_word_semantic_hierarchy,
    'cluster-cohesion-plot': make_cluster_cohesion_plot,
    'cluster-outlier-scores': make_cluster_outlier_scores,
    'sentence-similarity-distribution': make_sentence_similarity_distribution,
    'cluster-label-similarity-heatmap': make_cluster_label_similarity_heatmap,
    'keyword-rarity-plot': make_keyword_rarity_plot,
    'cluster-keyword-bipartite-graph': make_cluster_keyword_bipartite_graph,
    'topic-world-map': make_topic_world_map,
    'semantic-gravity-well': make_semantic_gravity_well_plot,
    'cluster-fingerprint-radar': make_cluster_fingerprint_radar_chart,
    'corpus-topic-radar': make_corpus_topic_radar,
    'keyword-trajectory': make_keyword_trajectory_plot,
    'sentence-bridge-identifier': make_sentence_bridge_identifier_plot,
    'galaxy-map-constellations': make_galaxy_map_plot,
    'gravitational-force-network': make_gravitational_force_network,
    'cluster-aura-plot': make_cluster_aura_plot,
}
