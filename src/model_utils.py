"""
Model training and evaluation utilities.

Functions for building prediction features, walk-forward splits,
training multiple model types, and computing evaluation metrics.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_momentum_features(
    returns: pd.Series,
    windows: Tuple[int, ...] = (5, 10, 21, 63),
) -> pd.DataFrame:
    """Compute momentum and technical features for a single return series.

    Args:
        returns: Daily return series.
        windows: Lookback windows in days.

    Returns:
        DataFrame of features indexed by date.
    """
    feats = pd.DataFrame(index=returns.index)

    for w in windows:
        feats[f"ret_{w}d"] = returns.rolling(w).sum()

    feats["vol_21d"] = returns.rolling(21).std() * np.sqrt(252)
    feats["sharpe_63d"] = (
        returns.rolling(63).mean() / returns.rolling(63).std()
    ) * np.sqrt(252)

    # Drawdown from 252-day high
    cum = (1 + returns).cumprod()
    rolling_max = cum.rolling(252, min_periods=1).max()
    feats["drawdown_252d"] = (cum - rolling_max) / rolling_max

    # RSI 14-day
    feats["rsi_14d"] = _compute_rsi(returns, 14)

    return feats


def _compute_rsi(returns: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    gains = returns.clip(lower=0)
    losses = (-returns).clip(lower=0)
    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macro_features(
    macro_data: pd.DataFrame,
) -> pd.DataFrame:
    """Compute features from macro data (VIX, yields, S&P 500).

    Args:
        macro_data: DataFrame with macro indicators.

    Returns:
        DataFrame of macro features.
    """
    feats = pd.DataFrame(index=macro_data.index)

    if "VIX" in macro_data.columns:
        feats["vix_level"] = macro_data["VIX"]
        feats["vix_change_5d"] = macro_data["VIX"].pct_change(5)

    if "S&P 500" in macro_data.columns:
        sp_ret = macro_data["S&P 500"].pct_change()
        feats["sp500_ret_5d"] = sp_ret.rolling(5).sum()
        feats["sp500_ret_21d"] = sp_ret.rolling(21).sum()

    if "10Y Treasury Yield" in macro_data.columns and "3M Treasury Yield" in macro_data.columns:
        feats["yield_spread"] = (
            macro_data["10Y Treasury Yield"] - macro_data["3M Treasury Yield"]
        )

    return feats


# ============================================================================
# WALK-FORWARD SPLIT
# ============================================================================

def walk_forward_split(
    dates: pd.DatetimeIndex,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create chronological train/validation/test masks.

    Args:
        dates: Sorted date index.
        train_pct: Fraction of dates for training.
        val_pct: Fraction of dates for validation.

    Returns:
        Tuple of boolean masks (train_mask, val_mask, test_mask).
    """
    n = len(dates)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    logger.info("Split: train=%d, val=%d, test=%d",
                train_mask.sum(), val_mask.sum(), test_mask.sum())
    return train_mask, val_mask, test_mask


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    actual_returns: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute comprehensive regression evaluation metrics.

    Args:
        y_true: True target values.
        y_pred: Predicted values.
        actual_returns: Actual realized returns for Sharpe calculation.

    Returns:
        Dict of metric names to values.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Direction accuracy
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    direction_acc = np.mean(true_direction == pred_direction)

    # Information Coefficient (Spearman rank correlation)
    ic, ic_pval = sp_stats.spearmanr(y_pred, y_true)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "direction_accuracy": direction_acc,
        "ic": ic,
        "ic_pval": ic_pval,
    }

    # Strategy Sharpe: invest when prediction > 0, cash otherwise
    if actual_returns is not None:
        signal = np.where(y_pred > 0, 1.0, 0.0)
        strategy_returns = signal * actual_returns
        if strategy_returns.std() > 0:
            metrics["strategy_sharpe"] = (
                strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            )
        else:
            metrics["strategy_sharpe"] = 0.0

    return metrics


def compute_confidence_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence_width: np.ndarray,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Evaluate calibration: narrower confidence intervals should have lower MAE.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        confidence_width: Width of confidence interval for each prediction.
        n_quantiles: Number of quantile bins.

    Returns:
        DataFrame with quantile, mean_width, mae for each bin.
    """
    errors = np.abs(y_true - y_pred)
    quantile_labels = pd.qcut(confidence_width, n_quantiles, labels=False, duplicates="drop")

    results = []
    for q in sorted(np.unique(quantile_labels)):
        mask = quantile_labels == q
        results.append({
            "quantile": q,
            "mean_width": confidence_width[mask].mean(),
            "mae": errors[mask].mean(),
            "n_samples": mask.sum(),
        })

    return pd.DataFrame(results)
