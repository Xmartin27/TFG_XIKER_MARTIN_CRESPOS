# Sistema de Recomendación Personalizada de Carteras de Inversión

---

## Descripción

Sistema cuantitativo de recomendación de carteras que combina detección de regímenes de mercado (HMM), predicción de retornos mediante ensemble de modelos de machine learning y clasificación del perfil de riesgo del inversor para generar recomendaciones personalizadas adaptadas al contexto macroeconómico actual.

---

## Pipeline

```
NB01  Universo de activos       →  33 activos, 2007–2026, 4.861 días de negociación
NB02  Generación de carteras    →  4.000 carteras estables (muestreo Dirichlet)
NB03  Detección de regímenes    →  6 estados HMM (selección por BIC)
NB04  Selección de carteras     →  40 carteras representativas (K-Means + score compuesto)
NB05  Perfil de riesgo          →  Clasificador de 5 niveles (Reg. Logística, F1=0.977)
NB06a Predicción de crecimiento →  5 modelos + ensemble, horizonte 22 días, walk-forward
NB06b Predicción de volatilidad →  4 modelos (GARCH, EWMA, XGBoost, CatBoost)
NB07  Motor de recomendación    →  Utilidad Markowitz + derating HMM + cash buffer
NB08  Validación                →  Backtest 2007–2026, hit rate 64–67%, Sharpe 0.57–0.67
```

---

## Estructura del repositorio

```
tfg_xiker_code/
├── notebooks_final/
│   ├── 01_data_universe_v2.ipynb
│   ├── 02_portfolio_generation.ipynb
│   ├── 03_regime_detection_hmm.ipynb
│   ├── 04_portfolio_selection.ipynb
│   ├── 05_client_risk_profile.ipynb
│   ├── 06a_growth_prediction_v5.ipynb
│   ├── 06b_volatility_prediction_v4.ipynb
│   ├── 07_recommendation_v3.ipynb
│   └── 08_validation_v3.ipynb
├── src/
│   ├── nb01_data.py               # Descarga y limpieza de precios
│   ├── nb02_portfolios.py         # Generación y filtrado de carteras
│   ├── nb03_hmm.py                # Entrenamiento y etiquetado del HMM
│   ├── nb04_selection.py          # Clustering y selección de carteras
│   ├── nb05_risk_profile.py       # Clasificador de perfil de riesgo
│   ├── nb07_recommendation.py     # Motor de recomendación (utilidad + régimen)
│   └── nb08_validation.py         # Backtest walk-forward
├── data/processed/
│   ├── asset_universe.json                    # Universo de activos y clases
│   ├── hmm_state_labels.json                  # Etiquetas semánticas de regímenes
│   ├── risk_model_metrics.json                # Métricas del clasificador de riesgo
│   ├── selected_portfolios_ids.json           # IDs de las 40 carteras seleccionadas
│   └── selected_portfolios_ids_optimal_v4.json
├── requirements.txt
└── README.md
```

---

## Metodología destacada

**Anti-leakage**: el modelo de predicción (NB06) usa walk-forward con purging y embargo de 22 días para evitar cualquier filtración de información futura al entrenamiento.

**Motor de recomendación** (NB07): función de utilidad de Markowitz `U = E(R) − ½·A·σ²` con coeficiente de aversión al riesgo A ∈ {10, 6, 3, 1.5, 1} por nivel; derating automático cuando el HMM detecta régimen de crisis; cash buffer (cartera sintética RF) para perfiles conservadores.

**Validación end-to-end** (NB08): backtest sobre 19 años confirma el trade-off riesgo-retorno esperado.

| Perfil          | Vol predicha | Hit rate | Retorno anual | Sharpe |
|-----------------|:------------:|:--------:|:-------------:|:------:|
| Muy Conservador | 5.3 %        | 66.4 %   | 8.8 %         | 0.673  |
| Conservador     | 6.0 %        | 66.7 %   | 9.5 %         | 0.633  |
| Moderado        | 6.6 %        | 64.5 %   | 11.1 %        | 0.673  |
| Agresivo        | 7.0 %        | 64.4 %   | 9.8 %         | 0.570  |
| Muy Agresivo    | 7.1 %        | 64.1 %   | 10.9 %        | 0.625  |

---

## Instalación y ejecución

```bash
# 1. Crear entorno virtual
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate          # Linux / Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Abrir los notebooks
jupyter notebook notebooks_final/
```

Los notebooks deben ejecutarse en orden (NB01 → NB08). Cada uno lee los outputs del anterior desde `data/processed/` y `data/results/`.

---

## Datos

- **Fuente de precios**: Yahoo Finance (yfinance) — descarga automática en NB01
- **Encuesta de riesgo**: NFCS 2021 (National Financial Capability Study), 2.820 respuestas
- **Activos**: 33 instrumentos — renta variable USA, renta variable internacional, renta fija, ETFs sectoriales, materias primas y REITs
- **Periodo**: enero 2007 – abril 2026
