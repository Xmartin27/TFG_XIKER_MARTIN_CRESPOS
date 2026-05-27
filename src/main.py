"""
Main pipeline: Portfolio Generation → HMM Regime Detection → Clustering → Output.

This script implements the complete minimal pipeline as specified:
1. Portfolio Generation: 4000 random portfolios
2. HMM Regime Detection: Gaussian HMM with 10 hidden states
3. Portfolio Reduction: KMeans clustering 4000 → 30 representative portfolios
"""

import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.portfolio_generation import generate_portfolios
from src.hmm_model import build_features, fit_hmm, extract_state_probabilities
from src.clustering import cluster_portfolios, compute_pca_projection

# ============================================================================
# SETUP
# ============================================================================

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration from .env
N_PORTFOLIOS = int(os.getenv("N_PORTFOLIOS", 4000))
N_ASSETS_MIN = int(os.getenv("N_ASSETS_MIN", 5))
N_ASSETS_MAX = int(os.getenv("N_ASSETS_MAX", 9))
MIN_WEIGHT = float(os.getenv("MIN_WEIGHT", 0.01))
MAX_WEIGHT = float(os.getenv("MAX_WEIGHT", 0.25))
PORTFOLIO_SEED = int(os.getenv("RANDOM_SEED", 42))

HMM_N_STATES = int(os.getenv("HMM_N_STATES", 10))
HMM_COVARIANCE_TYPE = os.getenv("HMM_COVARIANCE_TYPE", "diag")
HMM_N_ITER = int(os.getenv("HMM_N_ITER", 1000))
HMM_SEED = int(os.getenv("RANDOM_SEED", 42))

N_CLUSTERS = int(os.getenv("N_CLUSTERS", 30))
KMEANS_N_INIT = int(os.getenv("KMEANS_N_INIT", 20))
KMEANS_MAX_ITER = int(os.getenv("KMEANS_MAX_ITER", 500))
CLUSTERING_SEED = int(os.getenv("CLUSTERING_RANDOM_SEED", 42))

DATA_RAW_PATH = Path(os.getenv("DATA_RAW_PATH", "data/raw"))
DATA_PROCESSED_PATH = Path(os.getenv("DATA_PROCESSED_PATH", "data/processed"))
DATA_OUTPUTS_PATH = Path(os.getenv("DATA_OUTPUTS_PATH", "data/outputs"))

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_indices_data() -> pd.DataFrame:
    """
    Load indices data from CSV and compute returns.
    
    Returns
    -------
    returns : pd.DataFrame
        Daily log returns, indexed by date, columns are assets.
    """
    csv_path = DATA_RAW_PATH / "indices_fixed_daily_2000_04032026.csv"
    
    logger.info(f"Loading indices data from {csv_path}...")
    
    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    logger.info(f"  Loaded prices: shape {prices.shape}")
    logger.info(f"  Date range: {prices.index.min()} to {prices.index.max()}")
    logger.info(f"  Assets: {prices.columns.tolist()}")
    
    # Compute log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    logger.info(f"  Computed log returns: shape {log_returns.shape}")
    
    # Check for NaNs
    na_count = log_returns.isna().sum().sum()
    logger.info(f"  NaNs in returns: {na_count}")
    
    if na_count > 0:
        log_returns = log_returns.dropna(how="any")
        logger.info(f"  After dropping NaNs: shape {log_returns.shape}")
    
    return log_returns


# ============================================================================
# STEP 2: GENERATE PORTFOLIOS
# ============================================================================

def run_portfolio_generation(returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate 4000 random portfolios.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns.
    
    Returns
    -------
    weights : pd.DataFrame
        Portfolio weights, shape (n_portfolios, n_assets).
    portfolio_returns : pd.DataFrame
        Portfolio returns, shape (n_dates, n_portfolios).
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 1: PORTFOLIO GENERATION (4000 portfolios)")
    logger.info("="*70)
    
    weights, portfolio_returns = generate_portfolios(
        returns=returns,
        n_portfolios=N_PORTFOLIOS,
        n_assets_min=N_ASSETS_MIN,
        n_assets_max=N_ASSETS_MAX,
        min_weight=MIN_WEIGHT,
        max_weight=MAX_WEIGHT,
        random_seed=PORTFOLIO_SEED,
    )
    
    # Validation
    assert np.allclose(weights.sum(axis=1), 1.0), "Weights should sum to 1"
    assert np.all(weights >= -1e-10), "All weights should be >= 0"
    assert weights.shape[0] == N_PORTFOLIOS, f"Expected {N_PORTFOLIOS} portfolios"
    
    logger.info(f"✓ Generated {weights.shape[0]} portfolios")
    logger.info(f"  Weights shape: {weights.shape}")
    logger.info(f"  Portfolio returns shape: {portfolio_returns.shape}")
    logger.info(f"  Mean weight per portfolio: {weights.mean().mean():.6f}")
    logger.info(f"  Min/Max portfolio return: {portfolio_returns.mean().min():.6f} / {portfolio_returns.mean().max():.6f}")
    logger.info(f"  Min/Max portfolio volatility: {portfolio_returns.std().min():.6f} / {portfolio_returns.std().max():.6f}")
    
    return weights, portfolio_returns


# ============================================================================
# STEP 3: HMM REGIME DETECTION
# ============================================================================

def run_hmm_detection(returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Fit Gaussian HMM with 10 hidden states.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns.
    
    Returns
    -------
    hidden_states : pd.Series
        Predicted hidden state at each time step.
    state_probs : pd.DataFrame
        State probabilities at each time step.
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 2: HMM REGIME DETECTION (10 hidden states)")
    logger.info("="*70)
    
    # Build features
    features, scaler = build_features(
        returns=returns,
        market_proxy=returns.columns[0],
        vol_window=10,
    )
    
    logger.info(f"  Built features: shape {features.shape}")
    
    # Fit HMM
    model, hidden_states, transition_matrix = fit_hmm(
        features=features,
        n_states=HMM_N_STATES,
        covariance_type=HMM_COVARIANCE_TYPE,
        n_iter=HMM_N_ITER,
        random_state=HMM_SEED,
    )
    
    logger.info(f"✓ HMM fitted with {HMM_N_STATES} states")
    
    # State distribution
    state_counts = hidden_states.value_counts().sort_index()
    logger.info(f"  State distribution:")
    for state, count in state_counts.items():
        pct = 100.0 * count / len(hidden_states)
        logger.info(f"    State {state}: {count:6d} ({pct:5.1f}%)")
    
    # Extract state probabilities
    state_probs = extract_state_probabilities(model, features)
    logger.info(f"  State probabilities shape: {state_probs.shape}")
    
    # Transition matrix
    logger.info(f"  Transition matrix:\n{transition_matrix.to_string()}")
    
    return hidden_states, state_probs


# ============================================================================
# STEP 4: PORTFOLIO CLUSTERING & SELECTION
# ============================================================================

def run_clustering(
    weights: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
) -> Tuple[np.ndarray, pd.Index, pd.DataFrame]:
    """
    Perform KMeans clustering: reduce 4000 → 30 portfolios.
    
    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights.
    portfolio_returns : pd.DataFrame
        Portfolio returns.
    
    Returns
    -------
    labels : np.ndarray
        Cluster assignment for each portfolio.
    representative_portfolios : pd.Index
        Index of selected representative portfolios.
    cluster_summary : pd.DataFrame
        Summary statistics for each cluster.
    """
    logger.info("\n" + "="*70)
    logger.info(f"STEP 3: PORTFOLIO CLUSTERING (4000 → {N_CLUSTERS} portfolios)")
    logger.info("="*70)
    
    labels, representative_portfolios, cluster_summary = cluster_portfolios(
        weights=weights,
        portfolio_returns=portfolio_returns,
        n_clusters=N_CLUSTERS,
        random_state=CLUSTERING_SEED,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
    )
    
    logger.info(f"✓ Clustering complete")
    logger.info(f"  Selected {len(representative_portfolios)} representative portfolios")
    logger.info(f"\n{cluster_summary.to_string(index=False)}")
    
    return labels, representative_portfolios, cluster_summary


# ============================================================================
# STEP 5: SAVE OUTPUTS
# ============================================================================

def save_outputs(
    weights: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    hidden_states: pd.Series,
    representative_portfolios: pd.Index,
    cluster_summary: pd.DataFrame,
) -> None:
    """
    Save pipeline outputs to disk.
    
    Parameters
    ----------
    weights : pd.DataFrame
        Full portfolio weights (4000).
    portfolio_returns : pd.DataFrame
        Full portfolio returns (4000).
    hidden_states : pd.Series
        HMM hidden states.
    representative_portfolios : pd.Index
        Index of representative portfolios (30).
    cluster_summary : pd.DataFrame
        Cluster summary statistics.
    """
    logger.info("\n" + "="*70)
    logger.info("SAVING OUTPUTS")
    logger.info("="*70)
    
    DATA_OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Save full data
    weights.to_parquet(DATA_OUTPUTS_PATH / "weights_matrix_full.parquet")
    logger.info(f"  Saved: weights_matrix_full.parquet ({weights.shape})")
    
    portfolio_returns.to_parquet(DATA_OUTPUTS_PATH / "portfolio_returns_full.parquet")
    logger.info(f"  Saved: portfolio_returns_full.parquet ({portfolio_returns.shape})")
    
    hidden_states.to_csv(DATA_OUTPUTS_PATH / "hidden_states.csv")
    logger.info(f"  Saved: hidden_states.csv ({len(hidden_states)} time steps)")
    
    # Save selected representative portfolios
    representative_weights = weights.loc[representative_portfolios]
    representative_weights.to_parquet(DATA_OUTPUTS_PATH / "weights_matrix_representatives.parquet")
    logger.info(f"  Saved: weights_matrix_representatives.parquet ({representative_weights.shape})")
    
    representative_returns = portfolio_returns[representative_portfolios]
    representative_returns.to_parquet(DATA_OUTPUTS_PATH / "portfolio_returns_representatives.parquet")
    logger.info(f"  Saved: portfolio_returns_representatives.parquet ({representative_returns.shape})")
    
    # Save cluster summary
    cluster_summary.to_csv(DATA_OUTPUTS_PATH / "cluster_summary.csv", index=False)
    logger.info(f"  Saved: cluster_summary.csv")
    
    logger.info(f"\nAll outputs saved to {DATA_OUTPUTS_PATH}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the complete minimal pipeline."""
    logger.info("\n" + "█"*70)
    logger.info("█" + " "*68 + "█")
    logger.info("█" + "  PORTFOLIO OPTIMIZATION PIPELINE - MINIMAL VERSION".center(68) + "█")
    logger.info("█" + " "*68 + "█")
    logger.info("█"*70)
    
    # Step 1: Load data
    logger.info("\nLoading indices data...")
    returns = load_indices_data()
    
    # Step 2: Generate portfolios
    weights, portfolio_returns = run_portfolio_generation(returns)
    
    # Step 3: HMM regime detection
    hidden_states, state_probs = run_hmm_detection(returns)
    
    # Step 4: Clustering
    labels, representative_portfolios, cluster_summary = run_clustering(weights, portfolio_returns)
    
    # Step 5: Save outputs
    save_outputs(weights, portfolio_returns, hidden_states, representative_portfolios, cluster_summary)
    
    # Final summary
    logger.info("\n" + "█"*70)
    logger.info("█" + " "*68 + "█")
    logger.info("█" + "  PIPELINE COMPLETE ✓".center(68) + "█")
    logger.info("█" + " "*68 + "█")
    logger.info("█"*70 + "\n")
    
    logger.info("SUMMARY:")
    logger.info(f"  ✓ Portfolios generated: {len(weights)}")
    logger.info(f"  ✓ HMM hidden states: {HMM_N_STATES}")
    logger.info(f"  ✓ Final portfolios selected: {len(representative_portfolios)}")
    logger.info(f"  ✓ Outputs saved to: {DATA_OUTPUTS_PATH}")
    
    return weights, portfolio_returns, hidden_states, representative_portfolios


if __name__ == "__main__":
    main()
