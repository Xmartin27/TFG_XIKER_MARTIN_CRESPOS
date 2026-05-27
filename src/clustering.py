"""
Portfolio clustering and selection module.

Performs KMeans clustering on portfolio weights and select representative portfolios.
Reduces from N portfolios to K clusters, selecting best representative from each cluster.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


logger = logging.getLogger(__name__)


def cluster_portfolios(
    weights: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    n_clusters: int = 30,
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 500,
) -> Tuple[np.ndarray, pd.Index, pd.DataFrame]:
    """
    Perform KMeans clustering on portfolio weights and select representatives.

    Selection strategy: choose the portfolio closest to each cluster centroid.

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights, shape (n_portfolios, n_assets).
        Index is portfolio_id.
    portfolio_returns : pd.DataFrame
        Portfolio returns, shape (n_dates, n_portfolios).
        Columns match weights.index.
    n_clusters : int
        Number of clusters (final number of portfolios).
    random_state : int
        Random seed for reproducibility.
    n_init : int
        Number of initializations for KMeans.
    max_iter : int
        Maximum iterations for KMeans.

    Returns
    -------
    labels : np.ndarray
        Cluster assignment for each portfolio.
    representative_portfolios : pd.Index
        Index of selected representative portfolios.
    cluster_summary : pd.DataFrame
        Summary statistics for each cluster.
    """
    n_portfolios = len(weights)

    if n_clusters > n_portfolios:
        logger.warning(f"n_clusters ({n_clusters}) > n_portfolios ({n_portfolios}). Using n_clusters = {n_portfolios}")
        n_clusters = n_portfolios

    logger.info(f"Clustering {n_portfolios} portfolios into {n_clusters} clusters...")

    # Standardize weights for clustering (each portfolio is a point in asset space)
    X = weights.values

    # Fit KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        verbose=0,
    )
    labels = kmeans.fit_predict(X)

    logger.info(f"KMeans converged in {kmeans.n_iter_} iterations")

    # Select representative from each cluster (closest to centroid)
    representative_idx = []
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_points = X[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_points) == 0:
            continue

        # Find portfolio closest to centroid
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        closest_idx_in_cluster = np.argmin(distances)
        closest_global_idx = cluster_indices[closest_idx_in_cluster]

        representative_idx.append(closest_global_idx)

    representative_portfolios = weights.index[representative_idx]

    logger.info(f"Selected {len(representative_portfolios)} representative portfolios")

    # Compute cluster summary
    cluster_summary = []
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_weights = weights[cluster_mask]
        
        # Get portfolios in this cluster
        cluster_port_returns = portfolio_returns.iloc[:, cluster_mask]
        
        cluster_stats = {
            "cluster_id": cluster_id,
            "n_portfolios": cluster_mask.sum(),
            "mean_weight_std": cluster_weights.std(axis=0).mean(),
            "mean_return": cluster_port_returns.mean().mean(),
            "mean_volatility": cluster_port_returns.std().mean(),
        }
        cluster_summary.append(cluster_stats)

    summary_df = pd.DataFrame(cluster_summary)

    logger.info(f"Cluster summary:\n{summary_df.to_string()}")

    return labels, representative_portfolios, summary_df


def compute_pca_projection(
    weights: pd.DataFrame,
    n_components: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Project portfolio weights onto principal components.

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights, shape (n_portfolios, n_assets).
    n_components : int
        Number of principal components.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    projection : pd.DataFrame
        PCA projection, shape (n_portfolios, n_components).
    """
    pca = PCA(n_components=min(n_components, weights.shape[1]), random_state=random_state)
    X = weights.values
    transformed = pca.fit_transform(X)

    projection = pd.DataFrame(
        transformed,
        index=weights.index,
        columns=[f"PC{i+1}" for i in range(transformed.shape[1])],
    )

    logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    logger.info(f"Cumulative explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    return projection
