import numpy as np
import logging
from difflib import SequenceMatcher
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

try:
    import hdbscan
except ImportError:
    hdbscan = None

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None


def detect_duplicates(items: Sequence[Any], threshold: float = 0.9) -> List[Dict[str, Any]]:
    """
    Detect near-duplicate text items with pairwise normalized string similarity.

    This lightweight helper is intended for unit-testable, dependency-free checks.
    Embedding-based clustering is handled separately by ``cluster_embeddings_hdbscan``.
    """
    if not items:
        return []
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    duplicates = []
    normalized_items = ["" if item is None else str(item).strip().casefold() for item in items]

    for left_idx in range(len(normalized_items)):
        left = normalized_items[left_idx]
        if not left:
            continue

        for right_idx in range(left_idx + 1, len(normalized_items)):
            right = normalized_items[right_idx]
            if not right:
                continue

            score = SequenceMatcher(None, left, right).ratio()
            if score >= threshold:
                duplicates.append({
                    "left_index": left_idx,
                    "right_index": right_idx,
                    "score": score,
                    "left": items[left_idx],
                    "right": items[right_idx],
                })

    return duplicates

def cluster_embeddings_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 2,  # Reduced from 3 to 2 for smaller result sets
    min_samples: int = 1,
    metric: str = 'cosine',
    cluster_selection_epsilon: float = 0.05,  # Reduced from 0.1 to 0.05 for tighter clustering
    algorithm: str = 'generic'
) -> np.ndarray:
    """
    Cluster embeddings using HDBSCAN algorithm with optimized parameters for Jina AI embeddings.
    
    Args:
        embeddings: Array of embeddings to cluster
        min_cluster_size: Minimum size of a cluster (reduced for search results)
        min_samples: Number of samples in a neighborhood for a point to be core
        metric: Distance metric to use
        cluster_selection_epsilon: Distance threshold for cluster selection (smaller = tighter clusters)
        algorithm: Algorithm to use for clustering
        
    Returns:
        Array of cluster labels
    """
    if embeddings.shape[0] < min_cluster_size:
        logger.warning(f"Not enough samples ({embeddings.shape[0]}) for clustering (min_cluster_size={min_cluster_size}). All points will be marked as noise.")
        return np.full(embeddings.shape[0], -1)
    
    # Adaptive parameters based on dataset size
    adaptive_min_cluster_size = max(2, min(min_cluster_size, embeddings.shape[0] // 3))
    adaptive_epsilon = cluster_selection_epsilon
    
    # For small datasets (typical search results), be more permissive
    if embeddings.shape[0] <= 10:
        adaptive_min_cluster_size = 2
        adaptive_epsilon = 0.02  # Even tighter for small sets
        
    logger.info(f"Running HDBSCAN with: min_cluster_size={adaptive_min_cluster_size}, min_samples={min_samples}, "
                f"metric='{metric}', cluster_selection_epsilon={adaptive_epsilon}, algorithm='{algorithm}'")
    
    if hdbscan is None:
        logger.warning("hdbscan is not installed; falling back to DBSCAN for clustering.")
        return dbscan_clusters(embeddings, eps=cluster_selection_epsilon or 0.1, min_samples=min_samples)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=adaptive_min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=adaptive_epsilon,
        algorithm=algorithm
    )
    
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # Log clustering results
    unique_labels = np.unique(cluster_labels)
    n_clusters = len([label for label in unique_labels if label >= 0])
    n_noise = np.sum(cluster_labels == -1)
    
    logger.info(f"HDBSCAN finished. Found {n_clusters} clusters, {n_noise} noise points")
    logger.info(f"Cluster labels: {unique_labels}")
    
    return cluster_labels

def dbscan_clusters(embeddings: np.ndarray,
                    eps: float = 0.1,
                    min_samples: int = 2) -> np.ndarray:
    """
    Cluster L2-normalized embeddings with DBSCAN.
    :return: Array of cluster labels
    """
    if DBSCAN is None:
        logger.warning("scikit-learn is not installed; marking all embeddings as noise.")
        return np.full(embeddings.shape[0], -1)

    return DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='euclidean'
    ).fit_predict(embeddings)
