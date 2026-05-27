"""
Recommendation engine with three utility formulations.

Implements personalized portfolio recommendations based on:
1. Volatility-based utility
2. Max Drawdown-based utility  
3. Tail Risk (CVaR)-based utility
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_portfolio_metrics(returns_df):
    """
    Compute key metrics for each portfolio from daily returns.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns with shape (n_days, n_portfolios)
    
    Returns
    -------
    pd.DataFrame
        Portfolio metrics
    """
    metrics = pd.DataFrame(index=returns_df.columns)
    
    # Annualized return (252 trading days)
    metrics['annual_return'] = returns_df.mean() * 252
    
    # Annualized volatility
    metrics['volatility'] = returns_df.std() * np.sqrt(252)
    
    # Sharpe ratio
    metrics['sharpe'] = metrics['annual_return'] / metrics['volatility']
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns_df).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    
    # CVaR at 95% confidence
    var_95 = returns_df.quantile(0.05)
    metrics['cvar_95'] = returns_df[returns_df <= var_95].mean()
    
    # Skewness and kurtosis
    metrics['skewness'] = returns_df.skew()
    metrics['kurtosis'] = returns_df.kurtosis()
    
    return metrics


def compute_utilities(portfolio_metrics, client_lambda):
    """
    Compute three utility scores for each client.
    
    Parameters
    ----------
    portfolio_metrics : pd.DataFrame
        Portfolio characteristics (must have annual_return, volatility, max_drawdown, cvar_95)
    client_lambda : array-like
        Risk penalty parameter for each client in [0, 1]
        Higher lambda = lower risk tolerance
    
    Returns
    -------
    dict of np.ndarray
        Dictionary with keys 'volatility', 'drawdown', 'tail_risk'
        Each array has shape (n_clients, n_portfolios)
    """
    n_clients = len(client_lambda)
    n_portfolios = len(portfolio_metrics)
    
    returns = portfolio_metrics['annual_return'].values
    volatility = portfolio_metrics['volatility'].values
    max_dd = np.abs(portfolio_metrics['max_drawdown'].values)
    cvar = np.abs(portfolio_metrics['cvar_95'].values)
    
    utilities = {
        'volatility': np.zeros((n_clients, n_portfolios)),
        'drawdown': np.zeros((n_clients, n_portfolios)),
        'tail_risk': np.zeros((n_clients, n_portfolios)),
    }
    
    for i in range(n_clients):
        lam = client_lambda[i]
        
        # Formulation 1: Volatility penalty
        utilities['volatility'][i, :] = returns - lam * volatility
        
        # Formulation 2: Max Drawdown penalty
        utilities['drawdown'][i, :] = returns - lam * max_dd
        
        # Formulation 3: Tail-risk penalty
        utilities['tail_risk'][i, :] = returns - lam * cvar
    
    return utilities


def get_top_recommendations(utilities, k=3):
    """
    Get top-k recommended portfolios for each client.
    
    Parameters
    ----------
    utilities : np.ndarray
        Shape (n_clients, n_portfolios)
    k : int
        Number of top recommendations
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (indices of top-k, scores of top-k)
    """
    top_indices = np.argsort(-utilities, axis=1)[:, :k]
    top_scores = np.take_along_axis(utilities, top_indices, axis=1)
    return top_indices, top_scores


def evaluate_monotonicity(recommendations, portfolio_metrics, client_risk_scores):
    """
    Evaluate monotonicity: higher-risk clients should get higher-risk portfolios.
    
    Returns
    -------
    float
        Correlation between client risk score and recommended portfolio volatility
    """
    recommended_volatility = portfolio_metrics.iloc[recommendations[:, 0]]['volatility'].values
    correlation = np.corrcoef(client_risk_scores, recommended_volatility)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def evaluate_portfolio_distribution(recommendations, portfolio_metrics, clients_df):
    """
    Analyze recommended portfolios by client risk level.
    
    Returns
    -------
    pd.DataFrame
        Summary statistics by risk bucket
    """
    results = []
    unique_buckets = sorted(clients_df['risk_bucket'].unique())
    
    for bucket in unique_buckets:
        mask = clients_df['risk_bucket'] == bucket
        rec_indices = recommendations[mask, 0]  # First recommendation
        
        avg_return = portfolio_metrics.iloc[rec_indices]['annual_return'].mean()
        avg_vol = portfolio_metrics.iloc[rec_indices]['volatility'].mean()
        avg_dd = portfolio_metrics.iloc[rec_indices]['max_drawdown'].mean()
        count = mask.sum()
        
        results.append({
            'risk_bucket': bucket,
            'n_clients': count,
            'avg_return': avg_return,
            'avg_volatility': avg_vol,
            'avg_max_dd': avg_dd,
        })
    
    return pd.DataFrame(results)


def evaluate_economic_coherence(recommendations, portfolio_metrics, clients_df):
    """
    Evaluate economic quality via Sharpe ratio by risk level.
    
    Returns
    -------
    pd.DataFrame
        Sharpe ratio by risk bucket
    """
    results = []
    unique_buckets = sorted(clients_df['risk_bucket'].unique())
    
    for bucket in unique_buckets:
        mask = clients_df['risk_bucket'] == bucket
        rec_indices = recommendations[mask, 0]
        avg_sharpe = portfolio_metrics.iloc[rec_indices]['sharpe'].mean()
        
        results.append({
            'risk_bucket': bucket,
            'avg_sharpe': avg_sharpe,
        })
    
    return pd.DataFrame(results)


def compute_recommendation_overlap(rec1, rec2):
    """
    Compute Jaccard similarity between two recommendation sets.
    
    Returns
    -------
    float
        Overlap proportion [0, 1]
    """
    overlap_counts = 0
    for i in range(len(rec1)):
        set1 = set(rec1[i, :])
        set2 = set(rec2[i, :])
        overlap = len(set1.intersection(set2))
        overlap_counts += overlap
    
    return overlap_counts / (len(rec1) * rec1.shape[1])


def create_evaluation_summary(utilities, portfolio_metrics, clients_df):
    """
    Create comprehensive evaluation of all three formulations.
    
    Returns
    -------
    pd.DataFrame
        Evaluation metrics for each formulation
    """
    formulations = ['volatility', 'drawdown', 'tail_risk']
    results = []
    
    for form in formulations:
        utils = utilities[form]
        rec_indices, _ = get_top_recommendations(utils, k=3)
        
        # Compute metrics
        mono = evaluate_monotonicity(rec_indices, portfolio_metrics, clients_df['risk_score'].values)
        dist = evaluate_portfolio_distribution(rec_indices, portfolio_metrics, clients_df)
        cohere = evaluate_economic_coherence(rec_indices, portfolio_metrics, clients_df)
        
        avg_sharpe = cohere['avg_sharpe'].mean()
        return_spread = dist['avg_return'].max() - dist['avg_return'].min()
        
        results.append({
            'Formulation': form.replace('_', ' ').title(),
            'Monotonicity': mono,
            'Avg_Sharpe': avg_sharpe,
            'Return_Spread': return_spread,
            'Recommendations': rec_indices,
            'Distribution': dist,
            'Coherence': cohere,
        })
    
    df = pd.DataFrame(results)
    
    # Normalize metrics to [0, 1]
    for col in ['Monotonicity', 'Avg_Sharpe', 'Return_Spread']:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            df[f'{col}_norm'] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[f'{col}_norm'] = 0.5
    
    # Composite score: 40% monotonicity, 40% Sharpe, 20% return spread
    df['Composite_Score'] = (
        df['Monotonicity_norm'] * 0.4 +
        df['Avg_Sharpe_norm'] * 0.4 +
        df['Return_Spread_norm'] * 0.2
    )
    
    return df
