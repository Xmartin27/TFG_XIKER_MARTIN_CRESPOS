"""
Portfolio generation module.

Generates N random portfolios with constraints:
- Weights sum to 1
- No short selling (all weights >= 0)
- Uses vectorized NumPy operations
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_portfolios(
    returns: pd.DataFrame,
    n_portfolios: int = 4000,
    n_assets_min: int = 5,
    n_assets_max: int = 9,
    min_weight: float = 0.01,
    max_weight: float = 0.25,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate N random portfolios with constraints.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns with shape (n_dates, n_assets).
        Index must be DatetimeIndex.
    n_portfolios : int
        Number of portfolios to generate.
    n_assets_min : int
        Minimum number of assets per portfolio.
    n_assets_max : int
        Maximum number of assets per portfolio.
    min_weight : float
        Minimum weight constraint per asset.
    max_weight : float
        Maximum weight constraint per asset.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    weights_matrix : pd.DataFrame
        Shape (n_portfolios, n_assets). Rows are portfolios, columns are assets.
        Index is portfolio_id (1 to n_portfolios).
    portfolio_returns : pd.DataFrame
        Shape (n_dates, n_portfolios). Rows are dates, columns are portfolios.
        Index matches returns.index.
    """
    rng = np.random.default_rng(random_seed)
    n_assets = len(returns.columns)
    asset_names = returns.columns.tolist()

    # Validate constraints
    if n_assets_min > n_assets:
        n_assets_min = n_assets
    if n_assets_max > n_assets:
        n_assets_max = n_assets
    if n_assets_min > n_assets_max:
        raise ValueError(f"n_assets_min ({n_assets_min}) > n_assets_max ({n_assets_max})")

    weights_list = []
    valid_count = 0

    logger.info(f"Generating {n_portfolios} portfolios from {n_assets} assets...")

    while valid_count < n_portfolios:
        # Randomly select n_assets for this portfolio
        n_assets_this = rng.integers(n_assets_min, n_assets_max + 1)
        selected_idx = rng.choice(n_assets, size=n_assets_this, replace=False)

        # Initialize weights from Dirichlet (automatically sums to 1)
        raw_weights = rng.dirichlet(alpha=np.ones(n_assets_this))

        # Apply min/max weight constraints iteratively
        weights = apply_weight_constraints(raw_weights, min_weight, max_weight)

        # Create full weight vector (0 for non-selected assets)
        full_weights = np.zeros(n_assets)
        full_weights[selected_idx] = weights

        # Verify constraints
        assert np.isclose(full_weights.sum(), 1.0, atol=1e-6), "Weights should sum to 1"
        assert np.all(full_weights >= -1e-10), "All weights should be >= 0"
        assert np.all(full_weights <= 1.0 + 1e-10), "All weights should be <= 1"

        weights_list.append(full_weights)
        valid_count += 1

    # Create weights matrix
    weights_matrix = pd.DataFrame(
        np.array(weights_list),
        index=[f"portfolio_{i+1:04d}" for i in range(n_portfolios)],
        columns=asset_names,
    )
    weights_matrix.index.name = "portfolio_id"

    logger.info(f"Generated {n_portfolios} portfolios")
    logger.info(f"Weights matrix shape: {weights_matrix.shape}")

    # Compute portfolio returns: returns @ weights.T
    # Shape: (n_dates, n_portfolios)
    returns_aligned = returns[asset_names]  # Ensure column order matches
    portfolio_returns = returns_aligned @ weights_matrix.T

    logger.info(f"Portfolio returns shape: {portfolio_returns.shape}")

    # Compute basic statistics
    stats = pd.DataFrame({
        "mean_return": portfolio_returns.mean(axis=0),
        "volatility": portfolio_returns.std(axis=0),
    })
    stats["sharpe"] = stats["mean_return"] / stats["volatility"].replace(0, np.nan)
    logger.info(f"Mean return range: {stats['mean_return'].min():.6f} to {stats['mean_return'].max():.6f}")
    logger.info(f"Volatility range: {stats['volatility'].min():.6f} to {stats['volatility'].max():.6f}")

    return weights_matrix, portfolio_returns


def apply_weight_constraints(
    weights: np.ndarray,
    min_weight: float = 0.01,
    max_weight: float = 0.25,
    max_iterations: int = 100,
) -> np.ndarray:
    """
    Apply min/max weight constraints via iterative redistribution.

    Ensures:
    - All non-zero weights are between min_weight and max_weight
    - Sum is exactly 1
    - No negative weights

    Parameters
    ----------
    weights : np.ndarray
        Initial weights (should sum to ~1).
    min_weight : float
        Minimum non-zero weight.
    max_weight : float
        Maximum weight.
    max_iterations : int
        Maximum iterations for constraint satisfaction.

    Returns
    -------
    weights : np.ndarray
        Constrained weights that sum to 1.
    """
    weights = np.clip(weights, 0.0, None)

    for iteration in range(max_iterations):
        # Clip to max
        clipped_max = np.clip(weights, None, max_weight)
        overflow = weights - clipped_max
        weights = clipped_max

        # If any weight is between 0 and min_weight, decide: 0 or min_weight
        in_middle = (weights > 0.0) & (weights < min_weight)
        if np.any(in_middle):
            # Redistribute: set to 0 and redistribute mass to others
            mass_to_redistribute = weights[in_middle].sum()
            weights[in_middle] = 0.0

            # Redistribute to non-zero weights that are below max
            non_zero = weights > 0.0
            available_for_increase = max_weight - weights[non_zero]
            if np.any(non_zero) and available_for_increase.sum() > 1e-10:
                increase = np.minimum(available_for_increase, mass_to_redistribute / np.sum(non_zero))
                weights[non_zero] += increase
                mass_to_redistribute -= increase.sum()

            # If still mass left, distribute equally to all non-zero
            if mass_to_redistribute > 1e-10 and np.any(non_zero):
                weights[non_zero] += mass_to_redistribute / np.sum(non_zero)

        # Renormalize
        total = weights.sum()
        if total > 1e-10:
            weights /= total
        else:
            # Edge case: all weights became 0
            n = len(weights)
            weights = np.ones(n) / n

        # Check convergence
        if not np.any(in_middle):
            break

    # Final normalization and enforcement
    weights = np.clip(weights, 0.0, max_weight)
    weights /= weights.sum()

    return weights
