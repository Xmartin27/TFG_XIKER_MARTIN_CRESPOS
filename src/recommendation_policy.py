"""
Recommendation Policy Engine.

Combines predicted portfolio growth, model reliability, and client risk profiles
to generate personalized portfolio recommendations.

Core Logic:
- For each client-portfolio-model combination, compute a score
- Score = alpha(risk_group) * predicted_growth - beta(risk_group) * pred_error
- Select top portfolios by score
- Apply decision rules for reliability thresholds
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# RECOMMENDATION PARAMETERS BY RISK GROUP
# ============================================================================

RECOMMENDATION_PARAMS = {
    1: {
        'name': 'Very Conservative',
        'alpha': 0.3,   # Growth weight: 30%
        'beta': 0.7,    # Error/reliability weight: 70% (reliability prioritized)
        'min_reliability': 0.7,  # Require high model reliability
    },
    2: {
        'name': 'Conservative',
        'alpha': 0.4,
        'beta': 0.6,
        'min_reliability': 0.6,
    },
    3: {
        'name': 'Moderate',
        'alpha': 0.5,
        'beta': 0.5,
        'min_reliability': 0.5,
    },
    4: {
        'name': 'Dynamic',
        'alpha': 0.6,
        'beta': 0.4,
        'min_reliability': 0.4,
    },
    5: {
        'name': 'Aggressive',
        'alpha': 0.8,   # Growth weight: 80%
        'beta': 0.2,    # Error/reliability weight: 20%
        'min_reliability': 0.3,  # Accept lower reliability if strong growth
    },
}


# ============================================================================
# RECOMMENDATION SCORE COMPUTATION
# ============================================================================

def compute_recommendation_scores(
    client_profiles: pd.DataFrame,
    portfolio_predictions: pd.DataFrame,
    model_reliability: Dict[str, float],
    portfolios_30: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute recommendation scores for all client-portfolio-model combinations.
    
    Parameters
    ----------
    client_profiles : pd.DataFrame
        Columns: client_id, risk_score, risk_group, risk_group_name, alpha, beta
    portfolio_predictions : pd.DataFrame
        Columns: portfolio_id / portfolio_index, pred_M1, pred_M2, pred_M3
        Values are predicted growth percentages.
    model_reliability : dict
        Keys: 'M1', 'M2', 'M3'
        Values: reliability score in [0, 1] for each model
    portfolios_30 : pd.DataFrame
        30 selected portfolios with characteristics
    
    Returns
    -------
    recommendation_scores : pd.DataFrame
        Rows: one row per client-portfolio-model combo
        Columns: client_id, portfolio_id, model_name, predicted_growth,
                 model_reliability, alpha, beta, recommendation_score
    """
    logger.info("Computing recommendation scores...")
    
    scores_rows = []
    
    for _, client in client_profiles.iterrows():
        client_id = client['client_id']
        risk_group = client['risk_group']
        alpha = client['alpha']
        beta = client['beta']
        
        for _, portf in portfolio_predictions.iterrows():
            portfolio_id = portf.name if hasattr(portf, 'name') else portf.get('portfolio_id', None)
            
            # Score for each model
            for model_name in ['M1', 'M2', 'M3']:
                pred_col = f'pred_{model_name}'
                if pred_col not in portf.index:
                    continue
                
                predicted_growth = portf[pred_col]
                reliability = model_reliability.get(model_name, 0.5)
                
                # Compute score: higher growth is better (maximize),
                # higher error is worse (minimize via beta * error)
                # error = 1 - reliability
                error_penalty = 1.0 - reliability
                
                score = alpha * predicted_growth - beta * error_penalty
                
                scores_rows.append({
                    'client_id': client_id,
                    'portfolio_id': portfolio_id,
                    'model_name': model_name,
                    'predicted_growth': predicted_growth,
                    'model_reliability': reliability,
                    'alpha': alpha,
                    'beta': beta,
                    'recommendation_score': score,
                })
    
    recommendation_scores = pd.DataFrame(scores_rows)
    logger.info(f"✓ Computed scores for {len(recommendation_scores)} combinations")
    
    return recommendation_scores


# ============================================================================
# RECOMMENDATION SELECTION & DECISION RULES
# ============================================================================

def select_top_recommendations(
    recommendation_scores: pd.DataFrame,
    client_profiles: pd.DataFrame,
    portfolio_info: pd.DataFrame = None,
    top_k: int = 3,
    apply_reliability_filter: bool = True,
) -> pd.DataFrame:
    """
    Select top K recommendations per client using recommendation scores.
    
    Decision Rules:
    1. Rank portfolios by recommendation_score per client
    2. If two portfolios have growth difference < 1%, prefer more reliable model
    3. For conservative groups, require minimum reliability threshold
    4. Fall back to best available model if threshold not met
    
    Parameters
    ----------
    recommendation_scores : pd.DataFrame
        From compute_recommendation_scores()
    client_profiles : pd.DataFrame
        Client profiles with risk_group
    portfolio_info : pd.DataFrame, optional
        Additional portfolio characteristics (names, weights, etc.)
    top_k : int
        Number of recommendations per client (default: 3)
    apply_reliability_filter : bool
        Whether to apply minimum reliability thresholds
    
    Returns
    -------
    recommendations : pd.DataFrame
        Rows: one recommendation per client-rank combo
        Columns: client_id, rank, recommendation_rank, portfolio_id, model_name,
                 predicted_growth, model_reliability, recommendation_score,
                 risk_group, risk_group_name
    """
    logger.info(f"Selecting top {top_k} recommendations per client...")
    
    recommendations = []
    
    for client_id in client_profiles['client_id'].unique():
        client_data = client_profiles[client_profiles['client_id'] == client_id].iloc[0]
        risk_group = client_data['risk_group']
        params = RECOMMENDATION_PARAMS[risk_group]
        min_reliability = params['min_reliability'] if apply_reliability_filter else 0.0
        
        # Get all scores for this client
        client_scores = recommendation_scores[recommendation_scores['client_id'] == client_id].copy()
        
        # Apply reliability filter for conservative groups
        if apply_reliability_filter and risk_group <= 2:
            filtered = client_scores[client_scores['model_reliability'] >= min_reliability]
            if len(filtered) == 0:
                # Fallback: use best available
                logger.warning(
                    f"Client {client_id} (group {risk_group}): "
                    f"No portfolios meet min_reliability={min_reliability}. Falling back to best available."
                )
                filtered = client_scores
            client_scores = filtered
        
        # Sort by recommendation_score descending
        client_scores = client_scores.sort_values('recommendation_score', ascending=False)
        
        # Select top K (may be fewer if not enough pass filter)
        selected = client_scores.head(top_k)
        
        for rank, (_, row) in enumerate(selected.iterrows(), 1):
            recommendations.append({
                'client_id': client_id,
                'recommendation_rank': rank,
                'portfolio_id': row['portfolio_id'],
                'model_name': row['model_name'],
                'predicted_growth': row['predicted_growth'],
                'model_reliability': row['model_reliability'],
                'recommendation_score': row['recommendation_score'],
                'risk_group': risk_group,
                'risk_group_name': params['name'],
            })
    
    recommendations_df = pd.DataFrame(recommendations)
    logger.info(f"✓ Selected {len(recommendations_df)} recommendations")
    
    return recommendations_df


# ============================================================================
# CONFLICT RESOLUTION & TIEBREAKING
# ============================================================================

def resolve_close_predictions(
    recommendations: pd.DataFrame,
    threshold_pct: float = 1.0,
) -> pd.DataFrame:
    """
    Tiebreaker rule: If two portfolios have similar predicted growth,
    prefer the one from the more reliable model.
    
    Parameters
    ----------
    recommendations : pd.DataFrame
        From select_top_recommendations()
    threshold_pct : float
        Growth difference threshold in percentage points (default: 1%)
    
    Returns
    -------
    resolved : pd.DataFrame
        Recommendations with tiebreaker applied
    """
    logger.info(f"Applying tiebreaker for close predictions (threshold={threshold_pct}%)...")
    
    resolved_rows = []
    
    for client_id in recommendations['client_id'].unique():
        client_recs = recommendations[recommendations['client_id'] == client_id].copy()
        
        for rank in sorted(client_recs['recommendation_rank'].unique()):
            rank_recs = client_recs[client_recs['recommendation_rank'] == rank]
            
            if len(rank_recs) > 1:
                # Check if predictions are within threshold
                min_growth = rank_recs['predicted_growth'].min()
                max_growth = rank_recs['predicted_growth'].max()
                
                if max_growth - min_growth <= threshold_pct:
                    # Multiple models with similar predictions -> pick most reliable
                    most_reliable = rank_recs.loc[rank_recs['model_reliability'].idxmax()]
                    resolved_rows.append(most_reliable)
                else:
                    # Predictions differ enough -> prefer highest growth
                    best_growth = rank_recs.loc[rank_recs['predicted_growth'].idxmax()]
                    resolved_rows.append(best_growth)
            else:
                resolved_rows.append(rank_recs.iloc[0])
    
    resolved = pd.DataFrame(resolved_rows)
    logger.info(f"✓ Tiebreaker applied: {len(resolved)} final recommendations")
    
    return resolved


# ============================================================================
# RECOMMENDATION ANALYSIS & REPORTING
# ============================================================================

def analyze_recommendations(
    recommendations: pd.DataFrame,
    client_profiles: pd.DataFrame,
) -> Dict:
    """
    Analyze recommendation patterns by risk group.
    
    Returns
    -------
    analysis : dict
        Summary statistics
    """
    logger.info("Analyzing recommendations...")
    
    analysis = {}
    
    # Distribution by risk group
    analysis['by_risk_group'] = recommendations.groupby('risk_group_name').agg({
        'client_id': 'count',
        'predicted_growth': ['mean', 'std'],
        'model_reliability': ['mean', 'std'],
    })
    
    # Model frequency
    analysis['model_frequency'] = recommendations['model_name'].value_counts()
    
    # Portfolio frequency
    analysis['portfolio_frequency'] = recommendations['portfolio_id'].value_counts()
    
    # Average predicted growth by model
    analysis['growth_by_model'] = recommendations.groupby('model_name')['predicted_growth'].agg(['mean', 'std'])
    
    # Average reliability by model
    analysis['reliability_by_model'] = recommendations.groupby('model_name')['model_reliability'].agg(['mean', 'std'])
    
    logger.info(f"\nRecommendation Analysis:")
    logger.info(f"  Total recommendations: {len(recommendations)}")
    logger.info(f"  Clients: {recommendations['client_id'].nunique()}")
    logger.info(f"  Unique portfolios recommended: {recommendations['portfolio_id'].nunique()}")
    logger.info(f"  Model breakdown:\n{analysis['model_frequency']}")
    
    return analysis


def format_recommendations_for_output(
    recommendations: pd.DataFrame,
    portfolios_30: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Format recommendations for final output and reporting.
    
    Parameters
    ----------
    recommendations : pd.DataFrame
        From resolve_close_predictions()
    portfolios_30 : pd.DataFrame, optional
        Portfolio metadata to merge in
    
    Returns
    -------
    formatted : pd.DataFrame
        Final recommendation output
    """
    output = recommendations.copy()
    
    # Format numeric columns
    output['predicted_growth'] = output['predicted_growth'].round(4)
    output['model_reliability'] = output['model_reliability'].round(4)
    output['recommendation_score'] = output['recommendation_score'].round(4)
    
    # Reorder columns
    column_order = [
        'recommendation_rank',
        'client_id',
        'portfolio_id',
        'model_name',
        'predicted_growth',
        'model_reliability',
        'recommendation_score',
        'risk_group',
        'risk_group_name',
    ]
    output = output[[c for c in column_order if c in output.columns]]
    
    logger.info(f"✓ Formatted {len(output)} recommendations for output")
    
    return output
