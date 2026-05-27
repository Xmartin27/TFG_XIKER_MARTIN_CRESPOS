"""
Recommendation Strategy Engine
==============================
Sistema de recomendación de carteras basado en:
- Precisión de 3 modelos de predicción (M1, M2, M3)
- 5 perfiles de riesgo de clientes (VeryConservative, Conservative, Moderate, Aggressive, VeryAggressive)
- Asignación inteligente: clientes agresivos → modelos menos precisos pero mayor crecimiento
                         clientes conservadores → modelos más precisos pero menor riesgo
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# ============================================================================
# DEFINICIÓN DE GRUPOS DE RIESGO Y ESTRATEGIA
# ============================================================================

RISK_GROUPS = {
    'VeryConservative': {
        'risk_tolerance': 0.1,      # 0-20%
        'min_precision': 0.95,       # Requiere 95%+ precisión
        'target_return': 0.02,       # Objetivo: 2% anual
        'max_volatility': 0.10,      # Máx 10% volatilidad
        'preferred_models': ['M3'],  # Modelos más precisos
    },
    'Conservative': {
        'risk_tolerance': 0.2,
        'min_precision': 0.80,       # Requiere 80%+ precisión
        'target_return': 0.04,       # Objetivo: 4% anual
        'max_volatility': 0.15,
        'preferred_models': ['M3', 'M2'],
    },
    'Moderate': {
        'risk_tolerance': 0.4,
        'min_precision': 0.65,       # Requiere 65%+ precisión
        'target_return': 0.07,       # Objetivo: 7% anual
        'max_volatility': 0.20,
        'preferred_models': ['M2', 'M1'],
    },
    'Aggressive': {
        'risk_tolerance': 0.6,
        'min_precision': 0.50,       # Requiere 50%+ precisión (menos exigente)
        'target_return': 0.12,       # Objetivo: 12% anual
        'max_volatility': 0.25,
        'preferred_models': ['M1'],
    },
    'VeryAggressive': {
        'risk_tolerance': 0.9,
        'min_precision': 0.40,       # Requiere 40%+ precisión (mínimo)
        'target_return': 0.18,       # Objetivo: 18% anual
        'max_volatility': 0.35,
        'preferred_models': ['M1'],
    }
}

# ============================================================================
# MODELOS DE PREDICCIÓN Y SUS CARACTERÍSTICAS
# ============================================================================

class PredictionModel:
    """Modelo de predicción con precisión y características"""
    
    def __init__(self, name: str, precision: float, bias: str = 'neutral'):
        """
        Args:
            name: Nombre del modelo (M1, M2, M3)
            precision: Precisión del modelo (0-1)
            bias: 'optimistic' (M1), 'neutral' (M2), 'pessimistic' (M3)
        """
        self.name = name
        self.precision = precision
        self.bias = bias
        self.error = 1.0 - precision
        
    def adjust_prediction(self, predicted_return: float) -> float:
        """Ajusta predicción según el sesgo del modelo"""
        if self.bias == 'optimistic':
            # M1 tiende a ser optimista (sobrestima)
            adjustment = 1.0 + (self.error * 0.3)
        elif self.bias == 'pessimistic':
            # M3 tiende a ser pesimista (subestima)
            adjustment = 1.0 - (self.error * 0.1)
        else:
            # M2 es neutral
            adjustment = 1.0
        
        return predicted_return * adjustment

class ModelEnsemble:
    """Conjunto de 3 modelos de predicción"""
    
    def __init__(self):
        # Modelos con diferentes precisiones (din´amicas según contexto)
        self.M1 = PredictionModel('M1', precision=0.65, bias='optimistic')  # Menos preciso, optimista
        self.M2 = PredictionModel('M2', precision=0.80, bias='neutral')      # Medio, neutral
        self.M3 = PredictionModel('M3', precision=0.95, bias='pessimistic')  # Muy preciso, pesimista
        
        self.models = [self.M1, self.M2, self.M3]
        
    def update_precisions(self, m1_prec: float, m2_prec: float, m3_prec: float):
        """Actualiza precisiones según el régimen de mercado"""
        self.M1.precision = m1_prec
        self.M1.error = 1.0 - m1_prec
        
        self.M2.precision = m2_prec
        self.M2.error = 1.0 - m2_prec
        
        self.M3.precision = m3_prec
        self.M3.error = 1.0 - m3_prec
        
    def predict_all(self, portfolio_returns: Dict[int, float]) -> Dict[str, Dict[int, float]]:
        """Predice retorno para todas las carteras con cada modelo"""
        predictions = {}
        
        for model in self.models:
            predictions[model.name] = {}
            for portfolio_id, ret in portfolio_returns.items():
                adjusted_ret = model.adjust_prediction(ret)
                predictions[model.name][portfolio_id] = adjusted_ret
        
        return predictions

# ============================================================================
# ASIGNACIÓN INTELIGENTE DE CARTERAS A CLIENTES
# ============================================================================

class RecommendationAssigner:
    """Asigna carteras a clientes según precisión y perfil de riesgo"""
    
    def __init__(self, ensemble: ModelEnsemble):
        self.ensemble = ensemble
        self.risk_groups = RISK_GROUPS
        
    def classify_client(self, risk_score: float) -> str:
        """
        Clasifica cliente en grupo de riesgo (0-1 → VeryConservative a VeryAggressive)
        
        risk_score: 0.0-1.0 donde 0=muy conservador, 1=muy agresivo
        """
        if risk_score < 0.20:
            return 'VeryConservative'
        elif risk_score < 0.40:
            return 'Conservative'
        elif risk_score < 0.60:
            return 'Moderate'
        elif risk_score < 0.80:
            return 'Aggressive'
        else:
            return 'VeryAggressive'
    
    def get_best_portfolio_for_client(
        self,
        client_risk_score: float,
        predictions: Dict[str, Dict[int, float]],
        portfolio_metrics: pd.DataFrame
    ) -> Tuple[int, str, float]:
        """
        Encuentra la mejor cartera para un cliente específico
        
        Estrategia:
        - Clientes conservadores: modelo M3 (95% precisión) → menor riesgo
        - Clientes moderados: modelo M2 (80% precisión) → balance
        - Clientes agresivos: modelo M1 (65% precisión) → mayor crecimiento
        
        Returns:
            (portfolio_id, model_name, predicted_return)
        """
        client_group = self.classify_client(client_risk_score)
        risk_profile = self.risk_groups[client_group]
        
        # Selecciona el modelo apropiado basado en preferencia del grupo
        model_name = risk_profile['preferred_models'][0]
        
        # Obtiene predicciones del modelo seleccionado
        model_predictions = predictions[model_name]
        
        # Filtra carteras que cumplan con criterios de riesgo
        valid_portfolios = {}
        for portfolio_id, predicted_return in model_predictions.items():
            # Verifica que la cartera exista en métricas
            if portfolio_id not in portfolio_metrics.index:
                continue
            
            portfolio_vol = portfolio_metrics.loc[portfolio_id, 'volatility'] if 'volatility' in portfolio_metrics.columns else 0.15
            
            # Acepta si volatilidad está dentro de límite
            if portfolio_vol <= risk_profile['max_volatility']:
                valid_portfolios[portfolio_id] = predicted_return
        
        if not valid_portfolios:
            # Si no hay carteras válidas, toma la mejor sin restricción
            valid_portfolios = model_predictions
        
        # Selecciona cartera con máximo crecimiento esperado
        best_portfolio = max(valid_portfolios, key=valid_portfolios.get)
        best_return = valid_portfolios[best_portfolio]
        
        return best_portfolio, model_name, best_return
    
    def recommend_all_clients(
        self,
        df_clients: pd.DataFrame,
        predictions: Dict[str, Dict[int, float]],
        portfolio_metrics: pd.DataFrame
    ) -> pd.DataFrame:
        """Genera recomendaciones para todos los clientes"""
        recommendations = []
        
        for client_id, row in df_clients.iterrows():
            risk_score = row['risk_score']
            
            portfolio_id, model_name, predicted_return = self.get_best_portfolio_for_client(
                risk_score, predictions, portfolio_metrics
            )
            
            client_group = self.classify_client(risk_score)
            
            recommendations.append({
                'client_id': client_id,
                'risk_score': risk_score,
                'risk_group': client_group,
                'recommended_portfolio': portfolio_id,
                'model_used': model_name,
                'predicted_growth': predicted_return,
                'precision': getattr(getattr(self.ensemble, model_name), 'precision', 0.0)
            })
        
        return pd.DataFrame(recommendations)

# ============================================================================
# INTEGRACIÓN CON RÉGIMEN HMM
# ============================================================================

def adjust_model_precision_by_regime(
    hmm_state: int,
    hmm_state_chars: pd.DataFrame,
    base_precisions: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Ajusta precisión de modelos según régimen HMM
    
    - En mercados en BULL: M1 y M2 más precisos (sube) / M3 más conservador
    - En mercados en BEAR: M3 más preciso (bajos retornos)
    - En mercados en CRISIS: Todos menos precisos
    """
    
    if base_precisions is None:
        base_precisions = {'M1': 0.65, 'M2': 0.80, 'M3': 0.95}
    
    # Obtiene características del estado HMM
    if hmm_state in hmm_state_chars['state'].values:
        state_char = hmm_state_chars[hmm_state_chars['state'] == hmm_state].iloc[0]
        annual_return = state_char['avg_return_annual']
        annual_vol = state_char['avg_volatility_annual']
    else:
        annual_return = 0
        annual_vol = 0.20
    
    # Clasifica régimen
    if annual_return < -2.0 or annual_vol > 0.30:
        regime = 'CRISIS'
        adjustments = {'M1': -0.15, 'M2': -0.10, 'M3': -0.05}
    elif annual_return < 0.0 or annual_vol > 0.20:
        regime = 'BEAR'
        adjustments = {'M1': -0.05, 'M2': 0.0, 'M3': +0.05}
    elif annual_return > 2.0 and annual_vol < 0.20:
        regime = 'BULL'
        adjustments = {'M1': +0.10, 'M2': +0.05, 'M3': -0.05}
    else:
        regime = 'NEUTRAL'
        adjustments = {'M1': 0.0, 'M2': 0.0, 'M3': 0.0}
    
    # Aplica ajustes
    adjusted_precisions = {
        model: max(0.30, min(0.99, base_precisions[model] + adjustments[model]))
        for model in base_precisions.keys()
    }
    
    return adjusted_precisions, regime

# ============================================================================
# UTILIDADES DE REPORTEO
# ============================================================================

def generate_recommendation_report(
    recommendations_df: pd.DataFrame,
    predictions_metadata: Dict
) -> str:
    """Genera reporte de calidad de recomendaciones"""
    
    report = f"""
{'='*80}
RECOMMENDATION QUALITY REPORT
{'='*80}

📊 DISTRIBUTION BY RISK GROUP:
"""
    
    for group in RISK_GROUPS.keys():
        count = (recommendations_df['risk_group'] == group).sum()
        avg_growth = recommendations_df[recommendations_df['risk_group'] == group]['predicted_growth'].mean()
        avg_precision = recommendations_df[recommendations_df['risk_group'] == group]['precision'].mean()
        
        report += f"\n  {group:20s}: {count:5d} clients | Avg Growth: {avg_growth:6.2f}% | Avg Precision: {avg_precision:.1%}"
    
    report += f"\n\n{'='*80}"
    report += f"\n✓ Total recommendations generated: {len(recommendations_df):,}"
    report += f"\n✓ Average predicted growth: {recommendations_df['predicted_growth'].mean():.2f}%"
    report += f"\n✓ Average model precision: {recommendations_df['precision'].mean():.1%}"
    
    return report
