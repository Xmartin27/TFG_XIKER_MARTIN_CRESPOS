"""
Módulo de Evaluación de Métricas para TFG: Robo-Advisor Institucional
========================================================================

Senior Quant Developer Module: Evaluación exhaustiva de predicciones, regímenes y backtesting.

Este módulo proporciona funciones robustas para evaluar cada componente del pipeline:
- Clasificación de perfil de riesgo del cliente (ordinal)
- Predicciones de retornos de carteras
- Predicciones de volatilidad
- Detección de regímenes via HMM
- Backtesting final con costos de transacción reales

Todas las funciones incluyen manejo de NaNs, divisiones por cero y casos edge.
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, entropy
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error


# =============================================================================
# 1. EVALUACIÓN DEL PERFIL DE CLIENTE (Clasificación Ordinal 1-5)
# =============================================================================

def evaluate_risk_profiler(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    return_dict: bool = True
) -> Union[Dict[str, float], Tuple[float, float, float]]:
    """
    Evalúa el modelo de clasificación del perfil de riesgo del cliente.
    
    La clasificación es ORDINAL: los niveles de riesgo (1: Conservador, 5: Agresivo)
    tienen un orden implícito. Una predicción de "2" cuando es "1" es "menos mala"
    que predecir "5".
    
    MATEMÁTICA:
    -----------
    - Quadratic Weighted Kappa: Mide concordancia ponderada entre observador y predictor.
      Fórmula: kappa = 1 - (E_o / E_e) donde E_o es error observado, E_e es error esperado.
      Pesos cuadráticos: w_ij = (i - j)^2 penaliza errores mayores más fuertemente.
      
    - Macro F1-Score: Media no ponderada de F1-scores por cada clase (trata clases desequilibradas).
      Fórmula: F1 = 2 * (precision * recall) / (precision + recall)
      
    - MAE: Error Absoluto Medio. Interpretable en términos de "desviaciones de nivel de riesgo".
    
    Args:
        y_true: Array de verdaderos niveles de riesgo (valores: 1, 2, 3, 4, 5).
        y_pred: Array de predicciones del modelo.
        return_dict: Si True, retorna Dict[métrica → valor]. Si False, retorna tuple.
    
    Returns:
        Dict con claves: 'quadratic_kappa', 'macro_f1', 'mae' si return_dict=True.
        Caso contrario: (quadratic_kappa, macro_f1, mae)
    
    Raises:
        ValueError: Si y_true o y_pred contienen NaNs o tienen longitudes diferentes.
    """
    # Validación
    if len(y_true) != len(y_pred):
        raise ValueError(f"Longitudes diferentes: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("y_true o y_pred contienen NaNs. Limpia los datos antes de evaluar.")
    
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    # Quadratic Weighted Kappa (ordinal)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # Macro F1-Score (promedio no ponderado)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    if return_dict:
        return {
            'quadratic_kappa': qwk,
            'macro_f1': macro_f1,
            'mae': mae
        }
    else:
        return qwk, macro_f1, mae


# =============================================================================
# 2. EVALUACIÓN DE PREDICCIÓN DE RETORNOS
# =============================================================================

def evaluate_return_predictions(
    y_true_ret: np.ndarray,
    y_pred_ret: np.ndarray,
    return_dict: bool = True
) -> Union[Dict[str, float], Tuple[float, float, float]]:
    """
    Evalúa predicciones de retornos de carteras.
    
    MATEMÁTICA:
    -----------
    - Information Coefficient (IC): Correlación de rangos de Spearman entre predicciones
      y retornos realizados. Rango: [-1, +1].
      
      IC = ρ_Spearman(rank(y_pred_ret), rank(y_true_ret))
      
      Interpretación: IC > 0.05 indica capacidad predictiva débil pero consistente.
      En HFT, IC > 0.10 se considera excelente. En nuestro contexto (carteras), 
      IC > 0.03 es aceptable.
    
    - RMSE (Root Mean Squared Error): Penaliza errores grandes exponencialmente.
      
      RMSE = sqrt(mean((y_true - y_pred)^2))
      
      En contexto de retornos: RMSE en tanto-por-uno (ej: 0.02 = 2% error típico).
    
    - Hit Ratio: Precisión de dirección. ¿Acertamos si el retorno sube o baja?
      
      Hit Ratio = (# predicciones direccionalmente correctas) / (# total predicciones)
      
      Baseline aleatorio: 50%. Hit Ratio > 52% es estadísticamente significativo (t-test).
    
    Args:
        y_true_ret: Array de retornos reales (decimales, ej: 0.015 = 1.5%).
        y_pred_ret: Array de retornos predichos.
        return_dict: Si True, retorna diccionario. Si False, retorna tuple.
    
    Returns:
        Dict con claves: 'information_coefficient', 'rmse', 'hit_ratio'
    
    Raises:
        ValueError: Datos inválidos.
    """
    if len(y_true_ret) != len(y_pred_ret):
        raise ValueError(f"Longitudes diferentes: {len(y_true_ret)} vs {len(y_pred_ret)}")
    
    # Eliminar NaNs
    mask = ~(np.isnan(y_true_ret) | np.isnan(y_pred_ret))
    y_true_ret = y_true_ret[mask]
    y_pred_ret = y_pred_ret[mask]
    
    if len(y_true_ret) == 0:
        raise ValueError("No hay datos válidos después de eliminar NaNs.")
    
    # Information Coefficient (Spearman Rank Correlation)
    ic, _ = spearmanr(y_pred_ret, y_true_ret)
    ic = ic if not np.isnan(ic) else 0.0
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true_ret - y_pred_ret) ** 2))
    
    # Hit Ratio (dirección)
    y_true_sign = np.sign(y_true_ret)
    y_pred_sign = np.sign(y_pred_ret)
    hit_ratio = np.mean(y_true_sign == y_pred_sign)
    
    if return_dict:
        return {
            'information_coefficient': ic,
            'rmse': rmse,
            'hit_ratio': hit_ratio
        }
    else:
        return ic, rmse, hit_ratio


# =============================================================================
# 3. EVALUACIÓN DE PREDICCIÓN DE VOLATILIDAD
# =============================================================================

def evaluate_volatility_predictions(
    y_true_var: np.ndarray,
    y_pred_var: np.ndarray,
    return_dict: bool = True
) -> Union[Dict[str, float], Tuple[float, float]]:
    """
    Evalúa predicciones de volatilidad (varianza realizada vs predicha).
    
    MATEMÁTICA:
    -----------
    - QLIKE (Quasi-Likelihood Loss): Métrica estándar en literatura de forecasting de volatilidad.
      
      QLIKE = mean((y_true_var / y_pred_var) - log(y_true_var / y_pred_var) - 1)
      
      Propiedades:
        • Asimétrica: penaliza más las sub-predicciones (predecir baja volatilidad en realidad alta).
        • Escala invariante: no depende de unidades absolutas.
        • QLIKE = 0 ⟺ predicción perfecta.
        • QLIKE < 0 implica ratio < 1, sub-predicción.
        • QLIKE > 0 implica ratio > 1, sobre-predicción.
      
      En backtesting real, QLIKE < 0.01 se considera excelente.
    
    - MAE: Error Absoluto Medio en volatilidad (en unidades de desv. est. o varianza).
    
    Args:
        y_true_var: Array de varianzas/volatilidades realizadas.
        y_pred_var: Array de varianzas/volatilidades predichas.
        return_dict: Si True, retorna dict. Si False, retorna tuple.
    
    Returns:
        Dict con claves: 'qlike', 'mae_volatility'
    
    Raises:
        ValueError: Datos inválidos o con división por cero.
    """
    if len(y_true_var) != len(y_pred_var):
        raise ValueError(f"Longitudes diferentes: {len(y_true_var)} vs {len(y_pred_var)}")
    
    # Eliminar NaNs
    mask = ~(np.isnan(y_true_var) | np.isnan(y_pred_var))
    y_true_var = y_true_var[mask]
    y_pred_var = y_pred_var[mask]
    
    if len(y_true_var) == 0:
        raise ValueError("No hay datos válidos.")
    
    # Evitar división por cero o log(0)
    y_pred_var = np.maximum(y_pred_var, 1e-8)
    y_true_var = np.maximum(y_true_var, 1e-8)
    
    # QLIKE
    ratio = y_true_var / y_pred_var
    qlike = np.mean(ratio - np.log(ratio) - 1)
    
    # MAE
    mae_vol = mean_absolute_error(y_true_var, y_pred_var)
    
    if return_dict:
        return {
            'qlike': qlike,
            'mae_volatility': mae_vol
        }
    else:
        return qlike, mae_vol


# =============================================================================
# 4. EVALUACIÓN DE REGÍMENES HMM
# =============================================================================

def evaluate_hmm_regimes(
    returns: np.ndarray,
    hidden_states: np.ndarray,
    hmm_model_log_prob: float,
    n_params: int,
    return_dict: bool = True
) -> Union[Dict, Tuple]:
    """
    Evalúa la calidad de los regímenes detectados por HMM.
    
    MATEMÁTICA:
    -----------
    - AIC (Akaike Information Criterion):
      
      AIC = 2 * k - 2 * log(L)
      
      donde k = número de parámetros del modelo, L = verosimilitud máxima.
      Penaliza complejidad. Modelos con AIC menor son preferibles.
    
    - BIC (Bayesian Information Criterion):
      
      BIC = k * log(n) - 2 * log(L)
      
      donde n = número de observaciones. BIC penaliza más la complejidad que AIC
      (apropiado para datasets grandes como 15 años de datos diarios = ~3800 obs).
    
    - Separabilidad Financiera: Para cada régimen (detected hidden state),
      computa rentabilidad media anualizada y volatilidad anualizada.
      
      Si los regímenes son "buenos", debe haber separación clara:
        • Estado "Bull": retornos altos, volatilidad moderada
        • Estado "Bear": retornos bajos/negativos, volatilidad alta
        • Estado "Sideways": retornos neutros, volatilidad baja
    
    Args:
        returns: Array de retornos diarios (decimales).
        hidden_states: Array de estados HMM asignados a cada fecha.
        hmm_model_log_prob: Log-probabilidad máxima del modelo HMM fitted.
        n_params: Número total de parámetros del modelo (ej: para 3 estados: 
                  3*3 transiciones + 3 means + 3 variances + 3 weights - 1 = ~14).
        return_dict: Si True, retorna dict con AIC, BIC + DataFrame de separabilidad.
    
    Returns:
        Dict con claves:
            'aic': float
            'bic': float
            'regime_stats': pd.DataFrame con columnas [state, mean_return_ann, volatility_ann]
    """
    if len(returns) != len(hidden_states):
        raise ValueError(f"Longitudes diferentes: returns={len(returns)}, states={len(hidden_states)}")
    
    returns = np.asarray(returns)
    hidden_states = np.asarray(hidden_states, dtype=int)
    
    n_obs = len(returns)
    
    # AIC
    aic = 2 * n_params - 2 * hmm_model_log_prob
    
    # BIC
    bic = n_params * np.log(n_obs) - 2 * hmm_model_log_prob
    
    # Separabilidad financiera por régimen
    unique_states = np.unique(hidden_states)
    regime_stats_list = []
    
    for state in unique_states:
        mask = hidden_states == state
        state_returns = returns[mask]
        
        # Rentabilidad media anualizada (252 días trading/año)
        mean_ret_daily = np.nanmean(state_returns)
        mean_ret_ann = (1 + mean_ret_daily) ** 252 - 1
        
        # Volatilidad anualizada
        vol_daily = np.nanstd(state_returns)
        vol_ann = vol_daily * np.sqrt(252)
        
        regime_stats_list.append({
            'regime_id': state,
            'mean_return_annualized': mean_ret_ann,
            'volatility_annualized': vol_ann,
            'n_observations': np.sum(mask)
        })
    
    regime_stats_df = pd.DataFrame(regime_stats_list)
    
    if return_dict:
        return {
            'aic': aic,
            'bic': bic,
            'regime_statistics': regime_stats_df
        }
    else:
        return aic, bic, regime_stats_df


# =============================================================================
# 5. BACKTEST CON COSTOS DE TRANSACCIÓN
# =============================================================================

def backtest_client_portfolio(
    portfolio_returns: pd.DataFrame,
    recommended_weights: pd.DataFrame,
    initial_capital: float = 10000,
    transaction_cost: float = 0.001,
    risk_free_rate: float = 0.0,
    rebalance_freq: str = 'M',
    return_dict: bool = True
) -> Union[Dict, Tuple]:
    """
    Simula el crecimiento de un portafolio recomendado con costos de transacción reales.
    
    LÓGICA:
    -------
    1. Alinear `portfolio_returns` (T × N_assets) y `recommended_weights` (T × N_assets).
    2. Para cada fecha t:
       a) Retorno del portafolio = sum(weights[t] * returns[t])
       b) Calcular turnover = sum(|weights[t] - weights[t-1]|) / 2
       c) Restar coste de transacción: ret_neto = ret_bruto - turnover * tc
       d) Actualizar capital: capital[t] = capital[t-1] * (1 + ret_neto)
    3. Computar métricas finales.
    
    FÓRMULAS:
    ---------
    - Rentabilidad Anualizada: (valor_final / valor_inicial) ^ (252 / n_days) - 1
    
    - Maximum Drawdown: min(cumprod_returns - max_cumprod_hasta_esa_fecha) / max_cumprod
    
    - Sharpe Ratio: (ret_ann - rf_rate) / (volatility_ann)
      donde volatility_ann = std(retornos_diarios) * sqrt(252)
    
    - Portfolio Turnover Promedio: media de sum(|Δ weights|) a lo largo del tiempo.
    
    Args:
        portfolio_returns: DataFrame (T × N_assets) con retornos diarios de activos.
        recommended_weights: DataFrame (T × N_assets) con pesos recomendados (deben sumar ~1).
        initial_capital: Capital inicial (ej: 10000 €).
        transaction_cost: Comisión sobre turnover (ej: 0.001 = 10 bps = 0.1%).
        risk_free_rate: Tasa libre de riesgo anual para Sharpe (default 0).
        rebalance_freq: Frecuencia de rebalance ('M' = mensual, 'W' = semanal).
        return_dict: Si True, retorna Dict. Si False, retorna tuple.
    
    Returns:
        Dict con claves:
            'annualized_return': float
            'maximum_drawdown': float
            'sharpe_ratio': float
            'avg_turnover': float
            'final_capital': float
            'capital_evolution': pd.Series (valor del capital a lo largo del tiempo)
    
    Raises:
        ValueError: Mismatch en índices, NaNs, etc.
    """
    # Alineación de índices
    common_idx = portfolio_returns.index.intersection(recommended_weights.index)
    if len(common_idx) == 0:
        raise ValueError("No hay índices comunes entre portfolio_returns y recommended_weights.")
    
    returns = portfolio_returns.loc[common_idx].fillna(0).values  # T × N_assets
    weights = recommended_weights.loc[common_idx].fillna(1 / recommended_weights.shape[1]).values  # T × N_assets
    
    T = len(returns)
    
    if T < 2:
        raise ValueError("Se requieren al menos 2 observaciones.")
    
    # Normalizar pesos a cada fecha (suma = 1)
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # Inicializar arrays
    capital = np.zeros(T)
    capital[0] = initial_capital
    daily_returns_net = np.zeros(T)
    daily_returns_net[0] = 0  # primer día sin retorno
    
    turnover_list = []
    
    for t in range(1, T):
        # Retorno bruto del portafolio
        portfolio_ret_gross = np.nansum(weights[t] * returns[t])
        
        # Turnover (cambio en pesos)
        weight_change = np.abs(weights[t] - weights[t-1]).sum() / 2  # suma de cambios absolutos / 2
        turnover_list.append(weight_change)
        
        # Coste de transacción
        tc_cost = weight_change * transaction_cost
        
        # Retorno neto
        portfolio_ret_net = portfolio_ret_gross - tc_cost
        daily_returns_net[t] = portfolio_ret_net
        
        # Actualizar capital
        capital[t] = capital[t-1] * (1 + portfolio_ret_net)
    
    # Computar métricas finales
    final_capital = capital[-1]
    
    # Rentabilidad anualizada
    total_return = (final_capital - initial_capital) / initial_capital
    n_years = T / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Maximum Drawdown
    cumulative_capital = capital
    running_max = np.maximum.accumulate(cumulative_capital)
    drawdown = (cumulative_capital - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Sharpe Ratio
    daily_rets_nonzero = daily_returns_net[daily_returns_net != 0]
    if len(daily_rets_nonzero) > 1:
        volatility_daily = np.std(daily_rets_nonzero)
        volatility_annual = volatility_daily * np.sqrt(252)
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility_annual if volatility_annual > 0 else 0
    else:
        sharpe_ratio = 0.0
    
    # Turnover promedio
    avg_turnover = np.mean(turnover_list) if turnover_list else 0.0
    
    # Serie de capital a lo largo del tiempo
    capital_series = pd.Series(capital, index=common_idx, name='capital')
    
    if return_dict:
        return {
            'annualized_return': annualized_return,
            'maximum_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_turnover': avg_turnover,
            'final_capital': final_capital,
            'capital_evolution': capital_series,
            'total_return_simple': total_return
        }
    else:
        return annualized_return, max_drawdown, sharpe_ratio, avg_turnover


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def print_evaluation_report(
    risk_profile_metrics: Dict,
    return_metrics: Dict,
    volatility_metrics: Dict,
    hmm_metrics: Dict,
    backtest_metrics: Dict
) -> None:
    """
    Imprime un reporte formateado con todas las métricas de evaluación.
    Útil para presentación en consola y para la memoria del TFG.
    """
    print("\n" + "=" * 80)
    print("REPORTE EXHAUSTIVO DE EVALUACIÓN - ROBO-ADVISOR TFG")
    print("=" * 80)
    
    print("\n[1] EVALUACIÓN DEL PERFIL DE RIESGO DEL CLIENTE")
    print("-" * 80)
    for metric, value in risk_profile_metrics.items():
        print(f"   {metric:.<40} {value:.6f}")
    
    print("\n[2] EVALUACIÓN DE PREDICCIÓN DE RETORNOS")
    print("-" * 80)
    for metric, value in return_metrics.items():
        print(f"   {metric:.<40} {value:.6f}")
    
    print("\n[3] EVALUACIÓN DE PREDICCIÓN DE VOLATILIDAD")
    print("-" * 80)
    for metric, value in volatility_metrics.items():
        print(f"   {metric:.<40} {value:.6f}")
    
    print("\n[4] EVALUACIÓN DE REGÍMENES HMM")
    print("-" * 80)
    print(f"   AIC (Akaike Information Criterion).... {hmm_metrics['aic']:.2f}")
    print(f"   BIC (Bayesian Information Criterion).. {hmm_metrics['bic']:.2f}")
    print("\n   SEPARABILIDAD FINANCIERA POR RÉGIMEN:")
    print(hmm_metrics['regime_statistics'].to_string(index=False))
    
    print("\n[5] BACKTEST FINAL CON COSTOS DE TRANSACCIÓN")
    print("-" * 80)
    for metric, value in backtest_metrics.items():
        if metric != 'capital_evolution':
            if metric == 'annualized_return':
                print(f"   {metric:.<40} {value*100:>6.2f}%")
            elif metric == 'maximum_drawdown':
                print(f"   {metric:.<40} {value*100:>6.2f}%")
            elif metric == 'sharpe_ratio':
                print(f"   {metric:.<40} {value:>6.3f}")
            elif metric == 'avg_turnover':
                print(f"   {metric:.<40} {value*100:>6.2f}%")
            else:
                print(f"   {metric:.<40} {value:>10.2f}")
    
    print("\n" + "=" * 80 + "\n")
