#!/usr/bin/env python3
"""
nb07_metrics.py — Complete objective metrics for portfolio recommendations.

Implements:
- Sharpe Ratio (standard)
- Sortino Ratio (downside only)
- Calmar Ratio (recovery vs max DD)
- CVaR / Expected Shortfall
- Omega Ratio (probability of gain)
- Information Ratio (vs benchmark)

Usage:
    from src.nb07_metrics import compute_all_metrics, recommend_portfolios_v4_sharpe
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


# ─── CONSTANT ASSUMPTIONS ────────────────────────────────────────────────────

RF_RATE = 0.045  # Risk-free rate (annual, Treasury)
BENCHMARK_SHARPE = 0.60  # Benchmark Sharpe for Information Ratio comparison
TRADING_DAYS = 252


# ─── METRIC CALCULATIONS ─────────────────────────────────────────────────────

def compute_sharpe_ratio(
    returns: np.ndarray,
    rf_rate: float = RF_RATE,
    annual: bool = True,
) -> float:
    """
    Sharpe Ratio = (Mean Return - Rf) / Volatility
    
    Args:
        returns: Daily returns (decimal, not %)
        rf_rate: Annual risk-free rate
        annual: If True, annualize. Otherwise per-period.
    
    Returns:
        Sharpe ratio (annualized by default)
    """
    if len(returns) < 2:
        return np.nan
    
    mean_ret = np.mean(returns)
    vol = np.std(returns)
    
    if vol == 0:
        return np.nan
    
    sharpe = (mean_ret - rf_rate / TRADING_DAYS) / vol
    
    if annual:
        sharpe *= np.sqrt(TRADING_DAYS)
    
    return float(sharpe)


def compute_sortino_ratio(
    returns: np.ndarray,
    rf_rate: float = RF_RATE,
    target_return: float = 0.0,
    annual: bool = True,
) -> float:
    """
    Sortino Ratio = (Mean Return - Target) / Downside Deviation
    
    Only penalizes downside volatility (losses), ignores upside
    
    Args:
        returns: Daily returns
        rf_rate: Annual risk-free rate (used as target if target_return=0)
        target_return: Target return threshold (default: rf_rate / TRADING_DAYS)
        annual: Annualize result
    
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return np.nan
    
    if target_return == 0:
        target_return = rf_rate / TRADING_DAYS
    
    excess_ret = returns - target_return
    downside = excess_ret[excess_ret < 0]
    
    downside_vol = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 0
    
    if downside_vol == 0:
        return np.nan
    
    sortino = np.mean(excess_ret) / downside_vol
    
    if annual:
        sortino *= np.sqrt(TRADING_DAYS)
    
    return float(sortino)


def compute_calmar_ratio(
    returns: np.ndarray,
    annual: bool = True,
) -> float:
    """
    Calmar Ratio = Annualized Return / Max Drawdown (absolute)
    
    Measures recovery capacity: gain per unit of peak loss
    
    Args:
        returns: Daily returns
        annual: Annualize return (default True)
    
    Returns:
        Calmar ratio
    """
    if len(returns) < 2:
        return np.nan
    
    cum_ret = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - running_max) / running_max
    max_dd = np.min(drawdown)
    
    if max_dd >= 0:  # No drawdown
        return np.nan
    
    annual_ret = np.mean(returns) * TRADING_DAYS if annual else np.mean(returns)
    calmar = annual_ret / np.abs(max_dd)
    
    return float(calmar)


def compute_cvar(
    returns: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    CVaR / Expected Shortfall = Average of worst alpha% losses
    
    Args:
        returns: Daily returns
        alpha: Percentile threshold (default 5% = worst 5% of days)
    
    Returns:
        CVaR (as negative value indicating loss)
    """
    if len(returns) < 2:
        return np.nan
    
    threshold = np.percentile(returns, alpha * 100)
    worst_returns = returns[returns <= threshold]
    
    if len(worst_returns) == 0:
        return np.nan
    
    cvar = np.mean(worst_returns)
    return float(cvar)


def compute_omega_ratio(
    returns: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """
    Omega Ratio = Σ P(gain) × Gain / Σ P(loss) × |Loss|
    
    Probability of gain vs loss. Intuitive: Omega=1.5 means 1.5x more likely to gain
    
    Args:
        returns: Daily returns
        threshold: What counts as "gain" vs "loss" (default: 0 = positive vs negative)
    
    Returns:
        Omega ratio (can be any positive number)
    """
    if len(returns) < 2:
        return np.nan
    
    gains = returns[returns > threshold]
    losses = returns[returns <= threshold]
    
    if len(gains) == 0 or len(losses) == 0:
        return np.nan
    
    prob_gain = len(gains) / len(returns)
    prob_loss = len(losses) / len(returns)
    
    avg_gain = np.mean(gains)
    avg_loss = np.abs(np.mean(losses))
    
    if avg_loss == 0:
        return np.nan
    
    omega = (prob_gain * avg_gain) / (prob_loss * avg_loss)
    return float(omega)


def compute_information_ratio(
    returns: np.ndarray,
    benchmark_sharpe: float = BENCHMARK_SHARPE,
) -> float:
    """
    Information Ratio = (Portfolio Sharpe - Benchmark Sharpe) / Tracking Error
    
    Or approximated as: Alpha (excess return) / Tracking Error
    
    Args:
        returns: Daily returns
        benchmark_sharpe: Reference Sharpe (default 0.60 = market average)
    
    Returns:
        Information ratio (can be positive or negative)
    """
    if len(returns) < 2:
        return np.nan
    
    portfolio_sharpe = compute_sharpe_ratio(returns)
    
    if np.isnan(portfolio_sharpe):
        return np.nan
    
    ir = portfolio_sharpe - benchmark_sharpe
    return float(ir)


# ─── AGGREGATED METRIC COMPUTATION ───────────────────────────────────────────

def compute_all_metrics(
    returns_dict: Dict[str, np.ndarray],
    rf_rate: float = RF_RATE,
) -> pd.DataFrame:
    """
    Compute all 6 metrics for multiple return series.
    
    Args:
        returns_dict: {portfolio_id: daily_returns_array}
        rf_rate: Annual risk-free rate
    
    Returns:
        DataFrame with columns: sharpe, sortino, calmar, cvar, omega, ir
    """
    metrics_list = []
    
    for portfolio_id, returns in returns_dict.items():
        metrics = {
            'portfolio_id': portfolio_id,
            'sharpe': compute_sharpe_ratio(returns, rf_rate),
            'sortino': compute_sortino_ratio(returns, rf_rate),
            'calmar': compute_calmar_ratio(returns),
            'cvar_5pct': compute_cvar(returns, alpha=0.05),
            'omega': compute_omega_ratio(returns),
            'information_ratio': compute_information_ratio(returns),
        }
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)


# ─── SHARPE-OPTIMIZED RECOMMENDATION ALGORITHM ───────────────────────────────

def recommend_portfolios_v4_sharpe_optimized(
    df: pd.DataFrame,
    risk_level: int,
    historical_returns: Optional[Dict[int, np.ndarray]] = None,
    min_sharpe_thresholds: Dict[int, float] = None,
    lambda_diversif: float = 0.15,
    lambda_regime_defense: float = 0.10,
    top_n: int = 3,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Improved recommendation based on ABSOLUTE Sharpe Ratio + diversification.
    
    Algorithm:
    1. Calculate historical Sharpe for each portfolio
    2. Filter by minimum Sharpe threshold per RL
    3. Iteratively select top-3 with diversification penalty
    4. Return all 6 metrics for validation
    
    Args:
        df: Prediction frame from NB06 (with growth_pred, p_positive, sharpe, maxdd, etc)
        risk_level: 1-5
        historical_returns: {portfolio_id: daily_returns_array} for real Sharpe calculation
        min_sharpe_thresholds: Minimum acceptable Sharpe by RL
        lambda_diversif: Correlation penalty weight
        lambda_regime_defense: Drawdown penalty weight
        top_n: Number of recommendations (default 3)
    
    Returns:
        (recommendations_df, metrics_dict)
    """
    if min_sharpe_thresholds is None:
        min_sharpe_thresholds = {
            1: 0.80,  # Very Conservative
            2: 0.70,  # Conservative
            3: 0.60,  # Moderate
            4: 0.40,  # Aggressive
            5: 0.20,  # Very Aggressive
        }
    
    # 1. Calculate real Sharpe if historical returns available
    cands = df.copy()
    
    if historical_returns is not None:
        real_sharpes = {}
        for pid, rets in historical_returns.items():
            real_sharpes[pid] = compute_sharpe_ratio(rets)
        cands['real_sharpe'] = cands['portfolio_id'].map(real_sharpes)
    else:
        cands['real_sharpe'] = cands['sharpe']  # Fallback to predicted sharpe
    
    # 2. Filter by minimum Sharpe
    min_sharpe = min_sharpe_thresholds.get(risk_level, 0.50)
    cands_filtered = cands[cands['real_sharpe'] >= min_sharpe].copy()
    
    if len(cands_filtered) < top_n:
        # Fallback: use top-N by Sharpe if filter too strict
        cands_filtered = cands.nlargest(top_n, 'real_sharpe').copy()
    
    # 3. Iterative selection with diversification penalty
    selected_indices = []
    selected_ids = []
    
    for iteration in range(min(top_n, len(cands_filtered))):
        cands_available = cands_filtered.drop(selected_indices, errors='ignore').copy()
        
        if len(cands_available) == 0:
            break
        
        # Score based on Sharpe (primary) - diversification penalty
        cands_available['recommendation_score'] = (
            cands_available['real_sharpe']
            - lambda_diversif * len(selected_ids) / max(1, top_n)  # Increase penalty as we select
        )
        
        best_idx = cands_available['recommendation_score'].idxmax()
        selected_indices.append(best_idx)
        selected_ids.append(int(cands_filtered.loc[best_idx, 'portfolio_id']))
    
    # 4. Retrieve final recommendations
    result = cands_filtered.loc[selected_indices].copy()
    
    # 5. Calculate all metrics for result
    if historical_returns is not None:
        all_metrics = compute_all_metrics(historical_returns)
        result = result.merge(
            all_metrics[['portfolio_id', 'sortino', 'calmar', 'cvar_5pct', 'omega', 'information_ratio']],
            on='portfolio_id',
            how='left'
        )
    
    result['risk_level'] = risk_level
    
    metrics_summary = {
        'risk_level': risk_level,
        'selected_portfolios': selected_ids,
        'min_sharpe_threshold': min_sharpe,
        'mean_sharpe': result['real_sharpe'].mean(),
        'mean_sortino': result['sortino'].mean() if 'sortino' in result.columns else np.nan,
        'mean_calmar': result['calmar'].mean() if 'calmar' in result.columns else np.nan,
        'mean_omega': result['omega'].mean() if 'omega' in result.columns else np.nan,
    }
    
    return result, metrics_summary


# ─── VALIDATION FUNCTIONS ────────────────────────────────────────────────────

def validate_recommendation_quality(
    recs_df: pd.DataFrame,
    min_sharpe: Dict[int, float] = None,
) -> Dict:
    """
    Validate recommendation quality across all metrics.
    
    Returns:
        Dictionary with pass/fail for each check
    """
    if min_sharpe is None:
        min_sharpe = {1: 0.80, 2: 0.70, 3: 0.60, 4: 0.40, 5: 0.20}
    
    validation = {}
    
    for rl in sorted(recs_df['risk_level'].unique()):
        rl_data = recs_df[recs_df['risk_level'] == rl]
        
        validation[f'RL{rl}'] = {
            'n_portfolios': len(rl_data),
            'mean_sharpe': rl_data['real_sharpe'].mean(),
            'min_sharpe_met': (rl_data['real_sharpe'].min() >= min_sharpe.get(rl, 0.50)),
            'mean_sortino': rl_data['sortino'].mean() if 'sortino' in rl_data.columns else np.nan,
            'mean_calmar': rl_data['calmar'].mean() if 'calmar' in rl_data.columns else np.nan,
            'mean_omega': rl_data['omega'].mean() if 'omega' in rl_data.columns else np.nan,
        }
    
    return validation
