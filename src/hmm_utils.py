"""
Hidden Markov Model utilities for market regime detection.

Functions for building HMM features, selecting optimal number of states,
training the model, and generating regime features for downstream models.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def build_hmm_features(
    market_returns: pd.Series,
    vix: Optional[pd.Series] = None,
    vol_windows: Tuple[int, ...] = (21, 63),
) -> pd.DataFrame:
    """Build observable features for the HMM from market data.

    Args:
        market_returns: Daily log returns of market proxy (e.g., S&P 500).
        vix: Optional VIX level series.
        vol_windows: Rolling windows for realized volatility.

    Returns:
        DataFrame of HMM features aligned by date.
    """
    features = pd.DataFrame(index=market_returns.index)
    features["market_return"] = market_returns

    for w in vol_windows:
        features[f"realized_vol_{w}d"] = (
            market_returns.rolling(w).std() * np.sqrt(252)
        )

    if vix is not None and len(vix) > 0:
        features = features.join(vix.rename("vix"), how="left")
        features["vix"] = features["vix"].ffill()

    features = features.dropna()
    logger.info("HMM feature matrix: %s, features: %s",
                features.shape, list(features.columns))
    return features


def select_optimal_states(
    X_scaled: np.ndarray,
    state_range: Tuple[int, int] = (2, 6),
    n_inits: int = 10,
    n_iter: int = 300,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Train HMMs with different numbers of states and compare by BIC.

    Args:
        X_scaled: Standardized feature matrix (T x n_features).
        state_range: (min_states, max_states) inclusive.
        n_inits: Number of random initializations per state count.
        n_iter: Maximum EM iterations.
        random_seed: Base random seed.

    Returns:
        DataFrame with columns: n_states, log_likelihood, aic, bic, converged, model.
    """
    T, n_features = X_scaled.shape
    results = []

    for n in range(state_range[0], state_range[1] + 1):
        best_score = -np.inf
        best_model = None

        for init in range(n_inits):
            seed = random_seed + n * 100 + init
            try:
                model = GaussianHMM(
                    n_components=n,
                    covariance_type="full",
                    n_iter=n_iter,
                    random_state=seed,
                    tol=1e-4,
                )
                model.fit(X_scaled)
                score = model.score(X_scaled)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                logger.debug("n=%d, init=%d failed: %s", n, init, e)

        if best_model is not None:
            # Number of free parameters
            n_trans = n * (n - 1)  # transition matrix (rows sum to 1)
            n_means = n * n_features
            n_covars = n * n_features * (n_features + 1) // 2
            n_params = n_trans + n_means + n_covars + (n - 1)  # start probs

            aic = -2 * best_score + 2 * n_params
            bic = -2 * best_score + n_params * np.log(T)

            results.append({
                "n_states": n,
                "log_likelihood": best_score,
                "aic": aic,
                "bic": bic,
                "converged": best_model.monitor_.converged,
                "model": best_model,
            })
            logger.info("n=%d: LL=%.2f, BIC=%.2f, converged=%s",
                        n, best_score, bic, best_model.monitor_.converged)

    return pd.DataFrame(results)


def label_states(
    model: GaussianHMM,
    feature_names: List[str],
) -> Dict[int, str]:
    """Assign semantic labels to HMM states based on means.

    The state with the highest mean market return is labeled 'bull',
    the lowest is 'bear', and others are 'transition'.

    Args:
        model: Trained GaussianHMM.
        feature_names: Names of features (must include 'market_return').

    Returns:
        Dict mapping state index to label string.
    """
    ret_idx = feature_names.index("market_return") if "market_return" in feature_names else 0
    means = model.means_[:, ret_idx]
    n = model.n_components

    sorted_states = np.argsort(means)
    labels = {}
    labels[sorted_states[-1]] = "bull"
    labels[sorted_states[0]] = "bear"
    for s in sorted_states[1:-1]:
        labels[s] = "transition"

    if n == 2:
        labels[sorted_states[0]] = "bear"
        labels[sorted_states[1]] = "bull"

    return labels


def generate_regime_features(
    model: GaussianHMM,
    X_scaled: np.ndarray,
    dates: pd.DatetimeIndex,
    state_labels: Dict[int, str],
    forward_horizon: int = 21,
) -> pd.DataFrame:
    """Generate regime features for downstream predictive models.

    Args:
        model: Trained HMM.
        X_scaled: Standardized feature matrix.
        dates: Date index aligned with X_scaled.
        state_labels: Mapping from state index to semantic label.
        forward_horizon: Days ahead for predicted regime probabilities.

    Returns:
        DataFrame indexed by date with regime feature columns.
    """
    n_states = model.n_components
    states = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    df = pd.DataFrame(index=dates)
    df["current_regime"] = [state_labels.get(s, f"state_{s}") for s in states]
    df["current_regime_id"] = states

    # Soft probabilities for each labeled state (aggregated by label)
    unique_labels = set(state_labels.values())
    for label in unique_labels:
        state_ids = [sid for sid, lbl in state_labels.items() if lbl == label]
        df[f"prob_{label}"] = probs[:, state_ids].sum(axis=1)

    # Max probability = regime stability
    df["regime_stability"] = probs.max(axis=1)

    # Regime duration: consecutive days in current state
    duration = np.ones(len(states), dtype=int)
    for i in range(1, len(states)):
        if states[i] == states[i - 1]:
            duration[i] = duration[i - 1] + 1
    df["regime_duration"] = duration

    # Forward-predicted probabilities using transition matrix power
    trans_mat = model.transmat_
    trans_forward = np.linalg.matrix_power(trans_mat, forward_horizon)

    predicted_probs = probs @ trans_forward  # (T x n_states)
    for label in unique_labels:
        state_ids = [sid for sid, lbl in state_labels.items() if lbl == label]
        df[f"predicted_prob_{label}_{forward_horizon}d"] = predicted_probs[:, state_ids].sum(axis=1)

    return df
