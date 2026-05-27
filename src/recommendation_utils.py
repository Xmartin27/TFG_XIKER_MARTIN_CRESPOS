"""
Recommendation engine utilities.

Functions for mapping risk profiles to aversion parameters,
scoring portfolios, applying safety filters, and generating
personalized recommendations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ============================================================================
# RISK PROFILE MAPPING
# ============================================================================

RISK_LEVEL_CONFIG = {
    1: {"name": "Very Conservative", "lambda": 1.0},
    2: {"name": "Conservative", "lambda": 0.8},
    3: {"name": "Moderate", "lambda": 0.5},
    4: {"name": "Aggressive", "lambda": 0.25},
    5: {"name": "Very Aggressive", "lambda": 0.0},
}

# Safety filter thresholds by profile type
SAFETY_FILTERS = {
    "conservative": {  # lambda >= 0.8
        "min_p_positive": 0.70,
        "min_confidence_norm": 0.50,
        "min_direction_accuracy": 0.55,
    },
    "moderate": {  # 0.3 < lambda < 0.8
        "min_p_positive": 0.55,
        "min_confidence_norm": 0.0,
        "min_direction_accuracy": 0.50,
    },
    "aggressive": {  # lambda <= 0.3
        "min_p_positive": 0.40,
        "min_confidence_norm": 0.0,
        "min_direction_accuracy": 0.48,
    },
}


def get_filter_type(lam: float) -> str:
    """Map lambda to filter type string."""
    if lam >= 0.8:
        return "conservative"
    elif lam <= 0.3:
        return "aggressive"
    return "moderate"


# ============================================================================
# PORTFOLIO SCORING
# ============================================================================

def compute_portfolio_scores(
    predictions: pd.DataFrame,
    model_metrics: pd.DataFrame,
    lam: float,
) -> pd.DataFrame:
    """Score all candidate portfolios for a given risk aversion level.

    Args:
        predictions: DataFrame with columns:
            portfolio_id, predicted_return (mu), confidence_width (sigma_pred),
            model_name.
        model_metrics: DataFrame with columns:
            portfolio_id, model_name, direction_accuracy.
        lam: Risk aversion parameter (0=aggressive, 1=conservative).

    Returns:
        DataFrame with scoring columns added.
    """
    df = predictions.copy()

    # Prediction confidence ratio (like a prediction Sharpe)
    df["confidence"] = df["predicted_return"] / df["confidence_width"].clip(lower=1e-8)

    # Probability of positive return (assuming normal distribution)
    df["p_positive"] = norm.cdf(
        df["predicted_return"] / df["confidence_width"].clip(lower=1e-8)
    )

    # Normalize to [0, 1]
    mu_min, mu_max = df["predicted_return"].min(), df["predicted_return"].max()
    conf_min, conf_max = df["confidence"].min(), df["confidence"].max()

    if mu_max > mu_min:
        df["mu_normalized"] = (df["predicted_return"] - mu_min) / (mu_max - mu_min)
    else:
        df["mu_normalized"] = 0.5

    if conf_max > conf_min:
        df["confidence_normalized"] = (df["confidence"] - conf_min) / (conf_max - conf_min)
    else:
        df["confidence_normalized"] = 0.5

    # Score = lambda * confidence_normalized + (1 - lambda) * mu_normalized
    df["score"] = lam * df["confidence_normalized"] + (1 - lam) * df["mu_normalized"]

    # Merge direction accuracy from model metrics
    if model_metrics is not None and len(model_metrics) > 0:
        merge_cols = ["portfolio_id"]
        if "model_name" in model_metrics.columns and "model_name" in df.columns:
            merge_cols.append("model_name")
        df = df.merge(
            model_metrics[merge_cols + ["direction_accuracy"]],
            on=merge_cols,
            how="left",
        )
    else:
        df["direction_accuracy"] = 0.5

    return df


def apply_safety_filters(
    scored: pd.DataFrame,
    lam: float,
) -> pd.DataFrame:
    """Apply safety filters based on risk profile.

    Args:
        scored: DataFrame from compute_portfolio_scores.
        lam: Risk aversion parameter.

    Returns:
        Filtered DataFrame (only portfolios passing the filters).
    """
    filter_type = get_filter_type(lam)
    thresholds = SAFETY_FILTERS[filter_type]

    mask = (
        (scored["p_positive"] >= thresholds["min_p_positive"])
        & (scored["direction_accuracy"] >= thresholds["min_direction_accuracy"])
    )

    if thresholds["min_confidence_norm"] > 0:
        mask = mask & (scored["confidence_normalized"] >= thresholds["min_confidence_norm"])

    filtered = scored[mask].copy()

    if len(filtered) == 0:
        logger.warning(
            "No portfolios pass safety filters for lambda=%.2f (%s). "
            "Falling back to lowest-volatility portfolio.",
            lam, filter_type,
        )
        return pd.DataFrame()

    return filtered.sort_values("score", ascending=False)


def recommend_portfolios(
    client_risk_level: int,
    predictions: pd.DataFrame,
    model_metrics: pd.DataFrame,
    composition: pd.DataFrame,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """Generate top-N portfolio recommendations for a client.

    Args:
        client_risk_level: Integer 1-5.
        predictions: Prediction DataFrame.
        model_metrics: Model metrics DataFrame.
        composition: Portfolio composition DataFrame.
        top_n: Number of recommendations.

    Returns:
        List of recommendation dicts with portfolio details.
    """
    config = RISK_LEVEL_CONFIG[client_risk_level]
    lam = config["lambda"]
    profile_name = config["name"]

    scored = compute_portfolio_scores(predictions, model_metrics, lam)
    filtered = apply_safety_filters(scored, lam)

    if len(filtered) == 0:
        # Fallback: return the portfolio with lowest historical volatility
        logger.warning("Applying fallback for risk_level=%d", client_risk_level)
        if "annual_volatility" in predictions.columns:
            fallback = predictions.nsmallest(1, "annual_volatility")
        else:
            fallback = predictions.head(1)
        filtered = fallback

    top = filtered.head(top_n)
    recommendations = []

    for _, row in top.iterrows():
        pid = row["portfolio_id"]
        port_comp = composition[composition["portfolio_id"] == pid]
        tickers = port_comp["ticker"].tolist()
        weights = port_comp["weight"].tolist()

        rec = {
            "portfolio_id": pid,
            "tickers": tickers,
            "weights": weights,
            "predicted_return": row.get("predicted_return", np.nan),
            "confidence_interval_lower": row.get("predicted_return", 0) - row.get("confidence_width", 0),
            "confidence_interval_upper": row.get("predicted_return", 0) + row.get("confidence_width", 0),
            "p_positive": row.get("p_positive", np.nan),
            "score": row.get("score", np.nan),
            "model_name": row.get("model_name", "unknown"),
            "risk_level": client_risk_level,
            "risk_profile": profile_name,
            "direction_accuracy": row.get("direction_accuracy", np.nan),
        }

        # Textual justification
        rec["justification"] = _generate_justification(rec, lam)
        recommendations.append(rec)

    return recommendations


def _generate_justification(rec: Dict[str, Any], lam: float) -> str:
    """Generate a textual justification for a recommendation."""
    profile = rec["risk_profile"]
    mu = rec["predicted_return"]
    p_pos = rec["p_positive"]

    if lam >= 0.8:
        focus = "safety and prediction confidence"
    elif lam <= 0.3:
        focus = "expected growth potential"
    else:
        focus = "balanced risk-return tradeoff"

    return (
        f"Recommended for {profile} profile (focus: {focus}). "
        f"Predicted return: {mu:.2%}, probability of positive return: {p_pos:.1%}. "
        f"Portfolio contains {len(rec['tickers'])} assets with "
        f"model direction accuracy of {rec['direction_accuracy']:.1%}."
    )
