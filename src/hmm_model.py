"""
HMM regime detection module.

Implements Gaussian HMM with 10 hidden states to detect market regimes.
Uses simple feature engineering from asset returns.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def build_features(
    returns: pd.DataFrame,
    market_proxy: str = None,
    vol_window: int = 10,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Build standardized HMM features from market returns.

    Features:
    - Daily return of market proxy
    - Absolute daily return (volatility indicator)
    - Rolling volatility (std over vol_window)

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns indexed by date.
    market_proxy : str
        Column name to use as market proxy.
        If None, uses the first column or the average of all returns.
    vol_window : int
        Rolling window for volatility calculation.

    Returns
    -------
    features_scaled : pd.DataFrame
        Standardized features, shape (n_dates, 3).
    scaler : StandardScaler
        Fitted scaler for inverse transformation if needed.
    """
    if market_proxy is None or market_proxy not in returns.columns:
        # Use first asset or average across all
        if len(returns.columns) > 0:
            market_proxy = returns.columns[0]
            logger.warning(f"market_proxy not specified. Using {market_proxy}")
        else:
            raise ValueError("returns DataFrame is empty")

    # Extract market proxy returns
    market_returns = returns[market_proxy].values

    # Feature 1: Daily return
    f1 = market_returns.reshape(-1, 1)

    # Feature 2: Absolute return
    f2 = np.abs(market_returns).reshape(-1, 1)

    # Feature 3: Rolling volatility
    rolling_vol = pd.Series(market_returns).rolling(window=vol_window, min_periods=1).std().values
    rolling_vol = np.array(rolling_vol, copy=True, dtype=float)  # Make mutable copy
    rolling_vol[np.isnan(rolling_vol)] = 0.0
    f3 = rolling_vol.reshape(-1, 1)

    # Combine features
    features = np.hstack([f1, f2, f3])

    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Create DataFrame
    features_df = pd.DataFrame(
        features_scaled,
        index=returns.index,
        columns=["daily_return", "abs_return", "rolling_vol"],
    )

    logger.info(f"Built HMM features: shape {features_df.shape}")

    return features_df, scaler


def fit_hmm(
    features: pd.DataFrame,
    n_states: int = 10,
    covariance_type: str = "diag",
    n_iter: int = 1000,
    random_state: int = 42,
    tol: float = 1e-4,
) -> Tuple[GaussianHMM, pd.Series, pd.DataFrame]:
    """
    Fit a Gaussian HMM to detect market regimes.

    Parameters
    ----------
    features : pd.DataFrame
        Standardized features, shape (n_dates, n_features).
    n_states : int
        Number of hidden states for HMM.
    covariance_type : str
        Type of covariance matrix ('full', 'diag', 'spherical', 'tied').
    n_iter : int
        Maximum number of iterations for EM algorithm.
    random_state : int
        Random seed for reproducibility.
    tol : float
        Convergence tolerance for EM.

    Returns
    -------
    model : GaussianHMM
        Fitted HMM model.
    hidden_states : pd.Series
        Predicted hidden state at each time step.
    transition_matrix : pd.DataFrame
        State transition probability matrix.
    """
    X = features.values

    logger.info(f"Fitting Gaussian HMM with {n_states} states...")
    logger.info(f"  Covariance type: {covariance_type}")
    logger.info(f"  Input shape: {X.shape}")

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        tol=tol,
    )
    model.fit(X)

    # Predict hidden states
    hidden_states = model.predict(X)

    logger.info(f"HMM converged: {model.monitor_.converged}")
    logger.info(f"Final log-likelihood: {model.score(X):.4f}")

    # State statistics
    state_counts = np.bincount(hidden_states, minlength=n_states)
    logger.info(f"State distribution: {state_counts}")

    # Build transition matrix DataFrame
    transition_matrix = pd.DataFrame(
        model.transmat_,
        index=[f"State_{i}" for i in range(n_states)],
        columns=[f"State_{i}" for i in range(n_states)],
    )

    hidden_states_series = pd.Series(
        hidden_states,
        index=features.index,
        name="regime",
    )

    return model, hidden_states_series, transition_matrix


def extract_state_probabilities(
    model: GaussianHMM,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract predicted state probabilities for each time step.

    Parameters
    ----------
    model : GaussianHMM
        Fitted HMM model.
    features : pd.DataFrame
        Features with shape (n_dates, n_features).

    Returns
    -------
    state_probs : pd.DataFrame
        State probabilities, shape (n_dates, n_states).
        Index matches features.index.
    """
    X = features.values
    state_probs = model.predict_proba(X)

    state_probs_df = pd.DataFrame(
        state_probs,
        index=features.index,
        columns=[f"State_{i}_prob" for i in range(model.n_components)],
    )

    logger.info(f"Extracted state probabilities: shape {state_probs_df.shape}")

    return state_probs_df


def compute_regime_features(
    hidden_states: pd.Series,
    returns: pd.DataFrame = None,
    window: int = 20,
) -> pd.DataFrame:
    """
    Compute regime-conditioned features for each portfolio.

    For each date and state, compute statistics that will be used
    as features for downstream models.

    Parameters
    ----------
    hidden_states : pd.Series
        Hidden state at each time step, indexed by date.
    returns : pd.DataFrame, optional
        Portfolio or asset returns for computing state-specific statistics.
    window : int
        Rolling window for computing statistics per state.

    Returns
    -------
    regime_features : pd.DataFrame
        Features indexed by date.
    """
    # Simple regime features: indicator for each state presence
    n_states = hidden_states.max() + 1

    regime_features = pd.DataFrame(index=hidden_states.index)

    for state in range(n_states):
        # Indicator: 1 if in this state, 0 otherwise
        regime_features[f"in_state_{state}"] = (hidden_states == state).astype(int)

    logger.info(f"Computed regime features: shape {regime_features.shape}")

    return regime_features
