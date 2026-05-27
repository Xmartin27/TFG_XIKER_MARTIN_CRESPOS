"""
Portfolio generation and metrics utilities.

Functions for generating random portfolios with constraints,
computing portfolio return series, and calculating risk/return metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ============================================================================
# PORTFOLIO GENERATION
# ============================================================================

def generate_random_weights(
    n_assets: int,
    min_weight: float = 0.02,
    max_weight: float = 0.40,
    max_attempts: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a single random weight vector using Dirichlet distribution.

    Args:
        n_assets: Number of assets in the portfolio.
        min_weight: Minimum weight for any asset.
        max_weight: Maximum weight for any asset.
        max_attempts: Maximum re-sampling attempts.
        rng: NumPy random Generator instance.

    Returns:
        Array of weights summing to 1.0.
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(max_attempts):
        weights = rng.dirichlet(np.ones(n_assets))
        if np.all(weights >= min_weight) and np.all(weights <= max_weight):
            return weights

    # Fallback: clip and renormalize
    weights = rng.dirichlet(np.ones(n_assets))
    weights = np.clip(weights, min_weight, max_weight)
    weights /= weights.sum()
    return weights


def generate_portfolios(
    asset_tickers: List[str],
    n_portfolios: int = 4000,
    n_assets_range: Tuple[int, int] = (5, 15),
    min_weight: float = 0.02,
    max_weight: float = 0.40,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate random portfolio compositions.

    Args:
        asset_tickers: List of available asset ticker symbols.
        n_portfolios: Number of portfolios to generate.
        n_assets_range: (min, max) number of assets per portfolio.
        min_weight: Minimum weight per asset.
        max_weight: Maximum weight per asset.
        random_seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: portfolio_id, ticker, weight.
    """
    rng = np.random.default_rng(random_seed)
    all_tickers = np.array(asset_tickers)
    n_total = len(all_tickers)

    records = []
    for pid in range(n_portfolios):
        n_assets = rng.integers(n_assets_range[0], n_assets_range[1] + 1)
        n_assets = min(n_assets, n_total)
        selected = rng.choice(all_tickers, size=n_assets, replace=False)
        weights = generate_random_weights(n_assets, min_weight, max_weight, rng=rng)

        for ticker, w in zip(selected, weights):
            records.append({
                "portfolio_id": pid,
                "ticker": ticker,
                "weight": w,
            })

    df = pd.DataFrame(records)
    logger.info("Generated %d portfolios", n_portfolios)
    return df


def compute_portfolio_returns(
    composition: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute daily portfolio return series from composition and asset returns.

    Args:
        composition: DataFrame with columns (portfolio_id, ticker, weight).
        asset_returns: DataFrame of daily asset returns (dates x tickers).

    Returns:
        DataFrame of daily portfolio returns (dates x portfolio_id).
    """
    portfolio_ids = composition["portfolio_id"].unique()
    result = {}

    for pid in portfolio_ids:
        mask = composition["portfolio_id"] == pid
        tickers = composition.loc[mask, "ticker"].values
        weights = composition.loc[mask, "weight"].values

        # Only use tickers present in asset_returns
        valid = [t for t in tickers if t in asset_returns.columns]
        if len(valid) < len(tickers):
            missing = set(tickers) - set(valid)
            logger.warning("Portfolio %d: missing tickers %s", pid, missing)

        valid_idx = [i for i, t in enumerate(tickers) if t in asset_returns.columns]
        w = weights[valid_idx]
        w = w / w.sum()  # renormalize

        port_ret = (asset_returns[valid] * w).sum(axis=1)
        result[pid] = port_ret

    return pd.DataFrame(result)


# ============================================================================
# PORTFOLIO METRICS
# ============================================================================

def compute_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Compute maximum drawdown from a cumulative return series.

    Args:
        cumulative_returns: Series of cumulative returns (1 + r).cumprod().

    Returns:
        Maximum drawdown as a negative float (e.g., -0.25 = 25% drawdown).
    """
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()


def compute_portfolio_features(
    portfolio_returns: pd.DataFrame,
    market_returns: pd.Series,
    composition: pd.DataFrame,
    asset_class_map: Dict[str, str],
    broad_category_map: Dict[str, str],
    rf_annual: float = 0.0,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Compute a comprehensive feature vector for each portfolio.

    Args:
        portfolio_returns: Daily returns (dates x portfolio_id).
        market_returns: Daily returns of market benchmark (S&P 500).
        composition: Portfolio composition DataFrame.
        asset_class_map: Ticker -> asset class mapping.
        broad_category_map: Asset class -> broad category mapping.
        rf_annual: Annual risk-free rate.
        trading_days: Trading days per year.

    Returns:
        DataFrame with one row per portfolio and feature columns.
    """
    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    features = []

    for pid in portfolio_returns.columns:
        ret = portfolio_returns[pid].dropna()
        if len(ret) < trading_days:
            logger.warning("Portfolio %d has only %d days of returns", pid, len(ret))

        cum_ret = (1 + ret).cumprod()
        ann_return = ret.mean() * trading_days
        ann_vol = ret.std() * np.sqrt(trading_days)
        sharpe = (ann_return - rf_annual) / ann_vol if ann_vol > 0 else 0.0
        max_dd = compute_max_drawdown(cum_ret)

        # Downside metrics
        downside_ret = ret[ret < 0]
        downside_vol = downside_ret.std() * np.sqrt(trading_days) if len(downside_ret) > 0 else 1e-8
        sortino = (ann_return - rf_annual) / downside_vol if downside_vol > 0 else 0.0
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-8 else 0.0

        # Higher moments
        skewness = sp_stats.skew(ret.values)
        kurt = sp_stats.kurtosis(ret.values)

        # Beta vs market
        aligned = pd.concat([ret, market_returns], axis=1, join="inner").dropna()
        if len(aligned) > 30:
            cov_pm = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
            beta = cov_pm[0, 1] / cov_pm[1, 1] if cov_pm[1, 1] > 0 else 0.0
        else:
            beta = 0.0

        # Composition features
        port_comp = composition[composition["portfolio_id"] == pid]
        n_assets = len(port_comp)
        weights = port_comp["weight"].values
        hhi = np.sum(weights ** 2)

        # Asset class exposure
        broad_cats = {"Equity": 0.0, "Fixed Income": 0.0, "Commodities": 0.0, "REITs": 0.0}
        for _, row in port_comp.iterrows():
            tkr = row["ticker"]
            ac = asset_class_map.get(tkr, "Unknown")
            bc = broad_category_map.get(ac, "Other")
            if bc in broad_cats:
                broad_cats[bc] += row["weight"]

        # Intra-portfolio correlation
        port_tickers = port_comp["ticker"].tolist()
        # We skip this expensive computation here; it can be added if needed

        features.append({
            "portfolio_id": pid,
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "skewness": skewness,
            "kurtosis": kurt,
            "beta": beta,
            "n_assets": n_assets,
            "hhi": hhi,
            "pct_equity": broad_cats["Equity"],
            "pct_fixed_income": broad_cats["Fixed Income"],
            "pct_commodities": broad_cats["Commodities"],
            "pct_reits": broad_cats["REITs"],
        })

    return pd.DataFrame(features)
