"""
nb07_recommendation.py — Regime-aware portfolio recommendation engine.

All logic extracted from run_nb07.py so notebooks can call clean functions
with visible inputs/outputs and inline interpretations.
"""
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ── Regime danger calibration ─────────────────────────────────────────────────

STATE_BASE_RISK: Dict[int, float] = {
    0: 0.10,   # Sideways Low-Vol  (vol~7.5%,  Sharpe+0.20)
    1: 0.75,   # High Uncertainty  (vol~49.2%, Sharpe+0.14)
    2: 0.35,   # Transition        (vol~11.0%, Sharpe+0.07)
    3: 1.00,   # Crisis / Crash    (vol~43.0%, Sharpe-0.14)
    4: 0.30,   # Transition        (vol~14.3%, Sharpe+0.09)
    5: 0.45,   # Transition        (vol~22.2%, Sharpe+0.03)
}
MAX_PROB_TRANSITION = 0.1042   # Crisis-state self-persistence complement

# ── Risk profile parameters ───────────────────────────────────────────────────

RISK_PARAMS: Dict[int, dict] = {
    1: {"name": "Very Conservative", "min_p_pos": 0.55,
        "w_growth": 0.10, "w_prob": 0.40, "w_sharpe": 0.40, "w_dd": 0.10},
    2: {"name": "Conservative",      "min_p_pos": 0.53,
        "w_growth": 0.20, "w_prob": 0.35, "w_sharpe": 0.35, "w_dd": 0.10},
    3: {"name": "Moderate",          "min_p_pos": 0.51,
        "w_growth": 0.35, "w_prob": 0.25, "w_sharpe": 0.25, "w_dd": 0.15},
    4: {"name": "Aggressive",        "min_p_pos": 0.49,
        "w_growth": 0.50, "w_prob": 0.20, "w_sharpe": 0.20, "w_dd": 0.10},
    5: {"name": "Very Aggressive",   "min_p_pos": 0.45,
        "w_growth": 0.70, "w_prob": 0.10, "w_sharpe": 0.15, "w_dd": 0.05},
}

HORIZON_DAYS_DEFAULT = 22
RISK_AVERSION: Dict[int, float] = {1: 10.0, 2: 6.0, 3: 3.0, 4: 1.5, 5: 1.0}
W_UTILITY: Dict[int, float] = {1: 0.45, 2: 0.35, 3: 0.25, 4: 0.15, 5: 0.05}

CASH_PORTFOLIO_ID = 41
CASH_RF_ANNUAL = 0.02
CASH_VOL_ANNUAL = 0.01
CASH_P_POS = 0.98

# Target weights in crisis (w_growth zeroed, precision + protection maximized)
CRISIS_WEIGHTS: Dict[str, float] = {
    "w_growth": 0.00, "w_prob": 0.50, "w_sharpe": 0.35, "w_dd": 0.15,
}

# Predicted-vol weight per risk level (5th scoring dimension)
W_VOL: Dict[int, float] = {1: 0.20, 2: 0.15, 3: 0.10, 4: 0.05, 5: 0.02}

# MaxDD filter as percentile of full universe (0 = no filter)
# Raised the threshold for mid-risk to reduce exposure to extreme drawdowns
MAXDD_PCT: Dict[int, int] = {1: 70, 2: 60, 3: 60, 4: 40, 5: 20}

# M1 hit-rate threshold per HMM state for the regime filter
REGIME_HIT_THR: Dict[int, float] = {0: 0.45, 1: 0.45, 2: 0.50, 3: 0.55, 4: 0.50, 5: 0.50}
MIN_REGIME_SAMPLES = 10


# ── Regime context ────────────────────────────────────────────────────────────

def build_regime_context(
    regime_df: Optional[pd.DataFrame],
    state_labels_path: Optional[Path] = None,
) -> dict:
    """Compute regime_risk_score and build the regime_context dict.

    regime_risk_score ∈ [0, 1]:
        ~0.11 Sideways Low-Vol  |  ~0.25 Transition  |  ~0.83 Crisis

    Falls back to neutral defaults (rrs=0.35) if regime_df is None or missing columns.
    """
    if regime_df is None or "current_regime_id" not in regime_df.columns:
        print("WARNING: regime file not found — using neutral defaults (rrs=0.35)")
        return {
            "current_regime_id":    None,
            "current_regime_label": "Unknown",
            "regime_risk_score":    0.35,
            "prob_transition":      0.025,
            "prob_bear_22d":        0.0,
            "prob_bear_21d":        0.0,
            "is_crisis":            False,
            "is_bear_outlook":      False,
            "note":                 "hmm_regime_features.parquet not found; neutral defaults",
        }

    latest        = regime_df.iloc[-1]
    current_id    = int(latest["current_regime_id"])
    prob_trans    = float(latest.get("prob_transition", 0.025))
    prob_bear_22d = float(latest.get("predicted_prob_bear_22d", latest.get("predicted_prob_bear_21d", 0.0)))

    rrs = (
        0.50 * STATE_BASE_RISK.get(current_id, 0.35)
      + 0.30 * min(prob_trans / MAX_PROB_TRANSITION, 1.0)
    + 0.20 * prob_bear_22d
    )

    # Semantic label from JSON if available
    if state_labels_path is not None and Path(state_labels_path).exists():
        _labels = json.loads(Path(state_labels_path).read_text())
        current_label = _labels.get(str(current_id), latest.get("current_regime", "Unknown"))
    else:
        current_label = latest.get("current_regime", f"State {current_id}")

    is_crisis      = any(k in current_label for k in ["Crisis", "Crash"])
    is_bear_outlook = prob_bear_22d > 0.05

    ctx = {
        "current_regime_id":    current_id,
        "current_regime_label": current_label,
        "regime_risk_score":    float(rrs),
        "prob_transition":      float(prob_trans),
                "prob_bear_22d":        float(prob_bear_22d),
                "prob_bear_21d":        float(prob_bear_22d),
        "regime_stability":     float(latest.get("regime_stability", 1 - prob_trans)),
        "regime_duration_days": int(latest.get("regime_duration", 0)),
        "is_crisis":            bool(is_crisis),
        "is_bear_outlook":      bool(is_bear_outlook),
        "as_of_date":           str(regime_df.index[-1].date()),
    }
    print(f"Regime: '{current_label}' (state {current_id})  "
                    f"rrs={rrs:.3f}  prob_bear_22d={prob_bear_22d:.4f}")
    return ctx


# ── Volatility predictions ────────────────────────────────────────────────────

def load_pred_vol(
    results_dir: Path,
    model_priority: Tuple[str, ...] = ("CAT_vol", "XGB_vol", "LGB_vol", "Ensemble", "EGARCH", "EWMA"),
    processed_dir: Optional[Path] = None,
) -> Tuple[pd.Series, Optional[str]]:
    """Load annualized predicted volatility per portfolio from nb06 output.

    Priority:
    1. nb06 model predictions that cover the current portfolio universe (>= 10 portfolios).
    2. Historical annual volatility from portfolios_features.parquet (full coverage).

    Returns (ann_vol_series indexed by portfolio_id, model_name_used).
    """
    for fname in ["nb06v7_volatility_predictions.parquet",
                  "nb06_volatility_predictions.parquet",
                  "nb06v15_volatility_predictions.parquet",
                  "nb06v14_volatility_predictions.parquet"]:
        path = results_dir / fname
        if not path.exists():
            continue
        vol_df = pd.read_parquet(path)
        if "portfolio_id" not in vol_df.columns:
            continue
        last_fold = vol_df["fold"].max()
        for model in model_priority:
            sub = vol_df[(vol_df["model"] == model) & (vol_df["fold"] == last_fold)]
            if len(sub) == 0:
                continue
            tail = sub.sort_values("date").groupby("portfolio_id").tail(22)
            ann_vol = (
                tail.groupby("portfolio_id")["sigma2_pred"]
                    .mean()
                    .apply(lambda x: np.sqrt(252.0 * max(float(x), 0.0)))
            )
            if len(ann_vol) < 10:
                continue
            print(f"Vol predictions: model={model}, fold={last_fold}, "
                  f"n={len(ann_vol)}, range=[{ann_vol.min():.3f}, {ann_vol.max():.3f}]")
            return ann_vol.rename("pred_ann_vol"), model

    # Fallback: historical volatility from portfolios_features.parquet
    if processed_dir is None:
        processed_dir = results_dir.parent / "processed"
    feat_path = processed_dir / "portfolios_features.parquet"
    if feat_path.exists():
        feat = pd.read_parquet(feat_path)
        if "portfolio_id" in feat.columns and "annual_volatility" in feat.columns:
            ann_vol = feat.set_index("portfolio_id")["annual_volatility"].rename("pred_ann_vol")
            print(f"Vol (historical fallback): n={len(ann_vol)}, "
                  f"range=[{ann_vol.min():.3f}, {ann_vol.max():.3f}]")
            return ann_vol, "historical_vol"

    print("Vol predictions not available - scoring uses 4 dimensions only")
    return pd.Series(dtype=float), None


# ── Prediction frame ──────────────────────────────────────────────────────────

def build_pred_frame(
    model_results: pd.DataFrame,
    pred_vol_series: pd.Series,
    current_id: Optional[int],
) -> Tuple[pd.DataFrame, int]:
    """Build the scoring DataFrame from nb06 model results.

    Returns:
        pred_frame        — DataFrame indexed by portfolio_id
        regime_n_samples  — number of M1 samples in current regime (for filter gate)
    """
    _wanted = ["ENS_growth", "ENS_precision", "ENS_sharpe", "ENS_maxdd",
               "ENS_mape_pct", "ENS_brier_dir", "ENS_sortino", "M1_growth", "M2_growth"]
    pred_frame = model_results[
        [c for c in dict.fromkeys(_wanted) if c in model_results.columns]
    ].copy()
    pred_frame.index = pred_frame.index.astype(int)
    pred_frame["portfolio_id"] = pred_frame.index

    def _col(df, name, default):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    pred_frame["p_positive"]  = _col(pred_frame, "ENS_precision", 0.5)
    pred_frame["growth_pred"] = _col(pred_frame, "ENS_growth",    0.0)
    pred_frame["sharpe"]      = _col(pred_frame, "ENS_sharpe",    0.0)
    pred_frame["maxdd"]       = _col(pred_frame, "ENS_maxdd",     0.0)
    pred_frame["mape_pct"]    = _col(pred_frame, "ENS_mape_pct",  np.nan)

    pred_frame["pred_ann_vol"] = pred_vol_series.reindex(pred_frame.index)

    hit_col = f"M1_regime_{current_id}_hit" if current_id is not None else None
    n_col   = f"M1_regime_{current_id}_n"   if current_id is not None else None
    if hit_col and hit_col in model_results.columns:
        pred_frame["regime_hit"] = model_results[hit_col].reindex(pred_frame.index)
        regime_n_samples = int(model_results[n_col].mean()) if n_col in model_results.columns else 0
    else:
        pred_frame["regime_hit"] = np.nan
        regime_n_samples = 0

    return pred_frame, regime_n_samples


def _horizon_return_from_annual(rf_annual: float, horizon_days: int) -> float:
    return float((1.0 + rf_annual) ** (horizon_days / 252.0) - 1.0)


def _add_cash_candidate(
    df: pd.DataFrame,
    risk_level: int,
    regime_context: dict,
    horizon_days: int = HORIZON_DAYS_DEFAULT,
    rf_annual: float = CASH_RF_ANNUAL,
) -> Tuple[pd.DataFrame, bool]:
    """Append a synthetic cash portfolio for defensive profiles or crisis regimes."""
    if CASH_PORTFOLIO_ID in df.index:
        return df, False
    if not (risk_level <= 2 or regime_context.get("is_crisis") or regime_context.get("is_bear_outlook")):
        return df, False

    df = df.copy()
    cash_ret = _horizon_return_from_annual(rf_annual, horizon_days)
    row = {c: np.nan for c in df.columns}
    row.update({
        "portfolio_id": CASH_PORTFOLIO_ID,
        "p_positive": CASH_P_POS,
        "growth_pred": cash_ret,
        "sharpe": max(rf_annual / max(CASH_VOL_ANNUAL, 1e-6), 1.0),
        "maxdd": -0.01,
        "mape_pct": 0.0,
        "pred_ann_vol": CASH_VOL_ANNUAL,
    })
    df = pd.concat([df, pd.DataFrame([row]).set_index(pd.Index([CASH_PORTFOLIO_ID]))])
    df["portfolio_id"] = df.index.astype(int)
    return df, True


# ── Recommendation engine ─────────────────────────────────────────────────────

def recommend_portfolios(
    df: pd.DataFrame,
    risk_level: int,
    regime_context: dict,
    vol_available: bool,
    regime_n_samples: int = 0,
    top_n: int = 3,
) -> Tuple[pd.DataFrame, dict]:
    """Select top_n portfolios for a given risk level under the current regime.

    Mechanism:
        1. Effective weights: convex blend of base weights → crisis weights driven by rrs
        2. Optional 5th dimension: predicted volatility (penalizes high-vol portfolios)
        3. Filter A: minimum directional precision (tightened when rrs > 0.20)
        4. Filter B: MaxDD percentile gate (stricter in risky regimes)
        5. Filter C: M1 hit-rate in current HMM state (requires MIN_REGIME_SAMPLES)
        6. Staged fallback: C → B → A → full universe (never returns empty)
        7. Z-score weighted scoring

    Returns:
        recs             — top_n rows with scores and context columns
        effective_weights — dict of weights actually used
    """
    rrs    = regime_context["regime_risk_score"]
    cur_id = regime_context.get("current_regime_id")
    params = RISK_PARAMS[risk_level]

    # Convex blend: base → crisis as rrs increases
    eff = {k: (1 - rrs) * params[k] + rrs * CRISIS_WEIGHTS[k]
           for k in ["w_growth", "w_prob", "w_sharpe", "w_dd"]}

    # Add vol as 5th dimension (scales down other weights proportionally)
    if vol_available:
        w_vol = W_VOL[risk_level]
        eff   = {k: v * (1 - w_vol) for k, v in eff.items()}
        eff["w_vol"] = w_vol
    else:
        eff["w_vol"] = 0.0

    # Filter A: precision threshold
    p_boost   = max(0.0, 0.05 * (rrs - 0.20))
    min_p_pos = params["min_p_pos"] + p_boost
    fA = df[df["p_positive"] >= min_p_pos].copy()

    # Filter A2: minimum Sharpe (for risk-adjusted returns)
    min_sharpe = params.get("min_sharpe", 0.30)
    fA = fA[fA["sharpe"] >= min_sharpe].copy()

    # Filter B: MaxDD percentile
    dd_boost = 15.0 * max(0.0, (rrs - 0.20) / 0.80)
    eff_pct  = min(MAXDD_PCT[risk_level] + dd_boost, 95)
    if eff_pct > 0:
        thr_dd = np.percentile(df["maxdd"], eff_pct)
        fB = fA[fA["maxdd"] >= thr_dd].copy()
    else:
        fB = fA.copy()

    # Filter C: regime hit-rate gate
    fC = fB.copy()
    if (cur_id is not None and "regime_hit" in fC.columns
            and regime_n_samples >= MIN_REGIME_SAMPLES):
        thr_hit = REGIME_HIT_THR.get(cur_id, 0.50)
        fC = fB[fB["regime_hit"].fillna(0) >= thr_hit].copy()

    # Staged fallback
    cands = (fC if len(fC) >= top_n else
             fB if len(fB) >= top_n else
             fA if len(fA) >= top_n else df.copy())

    def zs(s):
        return (s - s.mean()) / (s.std() + 1e-9)

    cands = cands.copy()
    cands["score"] = (
        eff["w_growth"] * zs(cands["growth_pred"]) +
        eff["w_prob"]   * zs(cands["p_positive"])  +
        eff["w_sharpe"] * zs(cands["sharpe"])       +
        eff["w_dd"]     * zs(-cands["maxdd"].abs())
    )
    if vol_available and eff["w_vol"] > 0 and "pred_ann_vol" in cands.columns:
        fill_vol = cands["pred_ann_vol"].fillna(cands["pred_ann_vol"].mean())
        cands["score"] += eff["w_vol"] * zs(-fill_vol)

    for k, v in eff.items():
        cands[f"{k}_eff"] = round(float(v), 4)
    cands["regime_risk_score"] = round(rrs, 4)
    cands["regime_label"]      = regime_context["current_regime_label"]

    out_cols = ["portfolio_id", "score", "growth_pred", "p_positive", "sharpe",
                "maxdd", "pred_ann_vol", "regime_hit",
                "w_growth_eff", "w_prob_eff", "w_sharpe_eff", "w_dd_eff", "w_vol_eff",
                "regime_risk_score", "regime_label"]
    return (cands.nlargest(top_n, "score")
                 [[c for c in out_cols if c in cands.columns]]
                 .reset_index(drop=True),
            eff)


# ── Simplified regime-aware scoring (NB07 v2) ────────────────────────────────

LEVEL_NAMES_ES = {
    1: "Muy Conservador",
    2: "Conservador",
    3: "Moderado",
    4: "Agresivo",
    5: "Muy Agresivo",
}

# (w_safety, w_growth) base per risk level
# safety  = P(positive return) + low predicted vol  → "no perder"
# growth  = predicted return + precision              → "ganar"
RISK_BASE_W = {
    1: (0.80, 0.20),
    2: (0.65, 0.35),
    3: (0.50, 0.50),
    4: (0.30, 0.70),
    5: (0.10, 0.90),
}

# Minimum p_positive filter per risk level (loosened for aggressive)
# Slightly tightened for Risk 3 to improve hit-rate consistency across tiers
RISK_MIN_PRECISION = {1: 0.58, 2: 0.56, 3: 0.54, 4: 0.49, 5: 0.45}

# Regime shift: in bear → push everyone more defensive
REGIME_SAFE_SHIFT = {"bear": +0.20, "sideways": 0.00, "bull": -0.10}


def classify_regime_type(regime_context: dict) -> str:
    """Return 'bear', 'bull', or 'sideways' from the current regime context.

    Uses regime_risk_score (rrs) as the primary signal:
        rrs > 0.55 → bear   (crisis / high-vol / bearish outlook)
        rrs < 0.20 → bull   (calm / bull / low-vol)
        else       → sideways
    """
    rrs   = regime_context.get("regime_risk_score", 0.35)
    label = regime_context.get("current_regime_label", "").lower()

    if rrs > 0.55 or any(w in label for w in ["crisis", "crash", "bear", "high unc"]):
        return "bear"
    if rrs < 0.20 or any(w in label for w in ["bull", "low-vol", "alcista"]):
        return "bull"
    return "sideways"


def _normalise(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-9)


def compute_portfolio_scores(
    df: pd.DataFrame,
    risk_level: int,
    regime_type: str,
) -> pd.DataFrame:
    """Score each portfolio for a given risk level and market regime.

    safety_score  = 0.65·norm(p_positive) + 0.35·norm(1 − pred_ann_vol)
    growth_score  = 0.70·norm(growth_pred) + 0.30·norm(p_positive)
    score         = w_safe·safety_score + w_growth·growth_score

    Weights are blended with a regime shift so that in bear markets
    even aggressive clients get more defensiveness.
    """
    df = df.copy()

    # Safety component
    p_pos_n = _normalise(df["p_positive"])
    if "pred_ann_vol" in df.columns and df["pred_ann_vol"].notna().sum() > 0:
        vol_n = _normalise(1 - df["pred_ann_vol"].fillna(df["pred_ann_vol"].median()))
    else:
        vol_n = pd.Series(0.5, index=df.index)
    safety_score = 0.65 * p_pos_n + 0.35 * vol_n

    # Growth component
    growth_score = 0.70 * _normalise(df["growth_pred"]) + 0.30 * p_pos_n

    # Effective weights after regime shift
    w_safe_base, w_growth_base = RISK_BASE_W[risk_level]
    shift   = REGIME_SAFE_SHIFT.get(regime_type, 0.0)
    w_safe  = float(np.clip(w_safe_base + shift, 0.05, 0.95))
    w_growth = 1.0 - w_safe

    df["safety_score"] = safety_score.round(4)
    df["growth_score"] = growth_score.round(4)
    df["score"]        = (w_safe * safety_score + w_growth * growth_score).round(4)
    df["w_safe"]       = round(w_safe, 3)
    df["w_growth"]     = round(w_growth, 3)
    df["regime_type"]  = regime_type
    df["risk_level"]   = risk_level
    return df


def recommend_by_risk(
    pred_frame: pd.DataFrame,
    risk_level: int,
    regime_context: dict,
    top_n: int = 3,
) -> tuple:
    """Select top_n portfolios for a given risk level under the current regime.

    Steps:
    1. Classify current regime as bear / sideways / bull
    2. Score all portfolios with compute_portfolio_scores
    3. Apply a minimum precision filter (relaxed for aggressive profiles;
       also relaxed if too few candidates remain)
    4. Return top_n by score

    Returns:
        (recommendations_df, meta_dict)
    """
    regime_type = classify_regime_type(regime_context)
    df = compute_portfolio_scores(pred_frame, risk_level, regime_type)

    # Precision filter — staged fallback
    min_p = RISK_MIN_PRECISION[risk_level]
    candidates = df[df["p_positive"] >= min_p]
    if len(candidates) < top_n:
        min_p_relaxed = max(0.45, min_p - 0.05)
        candidates = df[df["p_positive"] >= min_p_relaxed]
    if len(candidates) < top_n:
        candidates = df  # last resort: use all

    # Tie-break toward higher-confidence portfolios so conservative tiers stay stable.
    top = (candidates
           .sort_values(
               by=["score", "p_positive", "direction_consensus", "model_agreement", "portfolio_id"],
               ascending=[False, False, False, False, True],
           )
           .head(top_n)
           .reset_index(drop=True))

    meta = {
        "risk_level":   risk_level,
        "level_name":   LEVEL_NAMES_ES[risk_level],
        "regime_type":  regime_type,
        "rrs":          round(float(regime_context.get("regime_risk_score", 0.35)), 3),
        "w_safe":       float(top["w_safe"].iloc[0]) if len(top) else 0.5,
        "w_growth":     float(top["w_growth"].iloc[0]) if len(top) else 0.5,
        "min_precision_used": min_p,
        "n_candidates": len(candidates),
    }
    return top, meta


# ── Output serialization ──────────────────────────────────────────────────────

def build_engine_json(
    pred_frame: pd.DataFrame,
    all_recs: Dict[int, pd.DataFrame],
    all_weights: Dict[int, dict],
    regime_context: dict,
    source_version: str,
    vol_model_used: Optional[str],
    vol_available: bool,
) -> dict:
    """Build the recommendation_engine.json dict (JSON-serializable)."""
    def _to_native(v):
        if hasattr(v, "item"): return v.item()
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v

    return {
        "last_update":       pd.Timestamp.now().isoformat(),
        "source_version":    source_version,
        "vol_model_used":    vol_model_used,
        "vol_available":     vol_available,
        "n_portfolios":      int(len(pred_frame)),
        "regime_context":    {k: _to_native(v) for k, v in regime_context.items()},
        "effective_weights": {str(rl): {k: round(float(v), 4) for k, v in w.items()}
                              for rl, w in all_weights.items()},
        "risk_params":       RISK_PARAMS,
        "recommendations":   {str(rl): r.to_dict(orient="records")
                              for rl, r in all_recs.items()},
    }


# ── V2: Vol-tier segmented recommendations ────────────────────────────────────
#
# Diseño: cada nivel de riesgo opera sobre un pool de candidatos distinto,
# definido por el percentil de volatilidad predicha.
#
#   Risk 1 — bottom 15% vol → candidatos "seguros" → ranking por p_positive
#   Risk 2 — bottom 38% vol → candidatos "conservadores"
#   Risk 3 — universo completo
#   Risk 4 — top 20% vol → candidatos "agresivos"
#   Risk 5 — top 30% vol → candidatos "máximo retorno" → ranking por growth_pred
#
# Esto garantiza:
#   - Carteras diferentes entre Risk 1 y Risk 5 (pools disjuntos)
#   - Para Risk 1: mayor hit rate (las predicciones son más fiables en vol baja)
#   - Para Risk 5: mayor retorno esperado, a costa de más error de predicción

VOL_TIER_QUANTILE: Dict[int, tuple] = {
    1: (0.00, 0.35),   # 35% menor vol  → Muy Conservador
    2: (0.00, 0.55),   # 55% menor vol  → Conservador
    3: (0.20, 0.75),   # universo amplio → Moderado
    4: (0.50, 1.00),   # 55% mayor vol  → Agresivo
    5: (0.65, 1.00),   # 35% mayor vol  → Muy Agresivo
}

# (w_safety, w_return) por nivel de riesgo
# safety_score = norm(p_positive)   → confianza del modelo = menos error
# return_score = norm(growth_pred)  → retorno esperado = más upside
RISK_V2_WEIGHTS: Dict[int, tuple] = {
    1: (0.99, 0.01),   # casi todo peso en confianza
    2: (0.98, 0.02),   # reforzar seguridad en R2
    3: (0.85, 0.15),   # prioridad a la seguridad para mejorar hit-rate
    4: (0.20, 0.80),   # más retorno en R4 para bajar su hit-rate relativo
    5: (0.00, 1.00),   # agresivo puro
}

LEVEL_NAMES_V2: Dict[int, str] = {
    1: "Muy Conservador (bajo error)",
    2: "Conservador",
    3: "Moderado",
    4: "Agresivo",
    5: "Muy Agresivo (máx. retorno)",
}


def _norm01(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def recommend_v2(
    pred_frame: pd.DataFrame,
    risk_level: int,
    regime_context: dict,
    top_n: int = 3,
) -> Tuple[pd.DataFrame, dict]:
    """Vol-tier segmented recommendation (NB07 v2).

    Pasos:
    1. Calcula cuantiles de vol sobre el universo completo (40 carteras).
    2. Filtra al tier de vol que corresponde al nivel de riesgo:
         Risk 1 → bottom 35% vol  (candidatos "predecibles", bajo error)
         Risk 5 → top 35% vol     (candidatos "agresivos", máx. retorno)
    3. Puntúa dentro del tier:
         score = w_safety * norm(p_positive) + w_return * norm(growth_pred)
         Risk 1: w_safety=0.90 → maximiza confianza (hit rate)
         Risk 5: w_return=0.90 → maximiza retorno esperado
    4. Ajuste por régimen: mercado bajista → empuja a todos hacia mayor seguridad.
    5. Devuelve top_n por score.
    """
    df  = pred_frame.copy()
    rrs = float(regime_context.get("regime_risk_score", 0.35))

    # Pesos base con ajuste por régimen
    w_safe_base, w_ret_base = RISK_V2_WEIGHTS[risk_level]
    if rrs > 0.55:          # bear: más defensivo
        w_safe = min(w_safe_base + 0.20, 0.95)
    elif rrs < 0.20:        # bull: más agresivo
        w_safe = max(w_safe_base - 0.10, 0.05)
    else:
        w_safe = w_safe_base
    w_ret = 1.0 - w_safe

    # Pre-filtro por tier de volatilidad predicha
    vol_filtered = False
    if "pred_ann_vol" in df.columns and df["pred_ann_vol"].notna().sum() >= top_n:
        q_lo, q_hi = VOL_TIER_QUANTILE[risk_level]
        vol_lo = df["pred_ann_vol"].quantile(q_lo) if q_lo > 0.0 else -np.inf
        vol_hi = df["pred_ann_vol"].quantile(q_hi) if q_hi < 1.0 else np.inf
        tier = df[(df["pred_ann_vol"] >= vol_lo) & (df["pred_ann_vol"] <= vol_hi)].copy()
        if len(tier) >= top_n:
            candidates = tier
            vol_filtered = True
        else:
            candidates = df.copy()
    else:
        candidates = df.copy()

    # Puntuación normalizada dentro del pool
    candidates = candidates.copy()
    p_pos_n  = _norm01(candidates["p_positive"])
    growth_n = _norm01(candidates["growth_pred"])

    # Safety score: acierto direccional + precisión de magnitud (1/MAPE)
    # Para conservadores, un MAPE bajo significa que cuando acertamos,
    # nos acercamos bastante al retorno real. Para agresivos esto importa menos.
    has_mape = ("mape_pct" in candidates.columns
                and candidates["mape_pct"].notna().sum() >= top_n
                and candidates["mape_pct"].gt(0).any())
    if has_mape:
        mape_filled = candidates["mape_pct"].fillna(candidates["mape_pct"].median())
        accuracy_n  = _norm01(1.0 / mape_filled)   # mayor accuracy_n = menor MAPE
        # safety_score: 55% dirección, 45% error de magnitud
        safety_score = (0.55 * p_pos_n + 0.45 * accuracy_n)
        candidates["accuracy_score"] = accuracy_n.round(4)
        mape_available = True
    else:
        safety_score = p_pos_n
        candidates["accuracy_score"] = np.nan
        mape_available = False

    candidates["safety_score"] = safety_score.round(4)
    candidates["return_score"] = growth_n.round(4)
    candidates["score"]        = (w_safe * safety_score + w_ret * growth_n).round(4)
    candidates["w_safe"]       = round(w_safe, 3)
    candidates["w_return"]     = round(w_ret, 3)
    candidates["risk_level"]   = risk_level

    top = candidates.nlargest(top_n, "score").reset_index(drop=True)

    regime_type = ("bear" if rrs > 0.55 else ("bull" if rrs < 0.20 else "sideways"))
    vol = candidates["pred_ann_vol"] if "pred_ann_vol" in candidates.columns else pd.Series(dtype=float)
    mape_col = candidates["mape_pct"] if "mape_pct" in candidates.columns else pd.Series(dtype=float)
    meta = {
        "risk_level":      risk_level,
        "level_name":      LEVEL_NAMES_V2[risk_level],
        "w_safe":          round(w_safe, 3),
        "w_return":        round(w_ret, 3),
        "vol_tier":        VOL_TIER_QUANTILE[risk_level],
        "vol_filtered":    vol_filtered,
        "n_candidates":    len(candidates),
        "mean_pred_vol":   round(float(vol.mean()), 4) if len(vol) > 0 else None,
        "mean_mape_pct":   round(float(mape_col.mean()), 4) if mape_available else None,
        "mape_in_safety":  mape_available,
        "regime_type":     regime_type,
        "rrs":             round(rrs, 3),
    }
    return top, meta


def recommend_v3(
    pred_frame: pd.DataFrame,
    risk_level: int,
    regime_context: dict,
    top_n: int = 3,
    horizon_days: int = HORIZON_DAYS_DEFAULT,
    include_cash: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Dynamic optimization: utility + HMM derating + cash buffer."""
    df = pred_frame.copy()
    cash_added = False
    if include_cash:
        df, cash_added = _add_cash_candidate(df, risk_level, regime_context, horizon_days)
    df["is_cash"] = (df.index.astype(int) == CASH_PORTFOLIO_ID)

    rrs = float(regime_context.get("regime_risk_score", 0.35))
    is_crisis = bool(regime_context.get("is_crisis") or regime_context.get("is_bear_outlook"))

    # Base weights with regime adjustment
    w_safe_base, w_ret_base = RISK_V2_WEIGHTS[risk_level]
    if is_crisis:
        w_safe = min(w_safe_base + (0.25 if risk_level <= 2 else 0.15), 0.95)
    elif rrs > 0.55:
        w_safe = min(w_safe_base + 0.20, 0.95)
    elif rrs < 0.20:
        w_safe = max(w_safe_base - 0.10, 0.05)
    else:
        w_safe = w_safe_base
    w_ret = 1.0 - w_safe

    # Vol tier filter (tightened in crisis for conservative profiles)
    vol_filtered = False
    if "pred_ann_vol" in df.columns and df["pred_ann_vol"].notna().sum() >= top_n:
        q_lo, q_hi = VOL_TIER_QUANTILE[risk_level]
        if is_crisis and risk_level <= 2:
            q_hi = max(q_lo + 0.10, q_hi - 0.10)
        vol_lo = df["pred_ann_vol"].quantile(q_lo) if q_lo > 0.0 else -np.inf
        vol_hi = df["pred_ann_vol"].quantile(q_hi) if q_hi < 1.0 else np.inf
        tier = df[(df["pred_ann_vol"] >= vol_lo) & (df["pred_ann_vol"] <= vol_hi)].copy()
        if len(tier) >= top_n:
            candidates = tier
            vol_filtered = True
        else:
            candidates = df.copy()
    else:
        candidates = df.copy()

    # Crisis derating: Sharpe + MaxDD filters (staged fallback)
    sharpe_filtered = False
    min_sharpe = 0.25
    if is_crisis:
        min_sharpe = 0.55 if risk_level <= 2 else 0.40
    elif rrs > 0.55:
        min_sharpe = 0.45 if risk_level <= 2 else 0.35
    if "sharpe" in candidates.columns and candidates["sharpe"].notna().sum() >= top_n:
        _f = candidates[candidates["sharpe"] >= min_sharpe]
        if len(_f) >= top_n:
            candidates = _f
            sharpe_filtered = True

    dd_quantile = None
    if "maxdd" in candidates.columns and candidates["maxdd"].notna().sum() >= top_n:
        base_q = 0.60 if risk_level == 1 else (0.50 if risk_level == 2 else 0.40)
        if is_crisis:
            base_q = min(base_q + 0.10, 0.90)
        thr_dd = candidates["maxdd"].quantile(base_q)
        _f = candidates[candidates["maxdd"] >= thr_dd]
        if len(_f) >= top_n:
            candidates = _f
            dd_quantile = base_q

    # Scores
    candidates = candidates.copy()
    p_pos_n  = _norm01(candidates["p_positive"])
    growth_n = _norm01(candidates["growth_pred"])

    has_mape = ("mape_pct" in candidates.columns
                and candidates["mape_pct"].notna().sum() >= top_n
                and candidates["mape_pct"].gt(0).any())
    if has_mape:
        mape_filled = candidates["mape_pct"].fillna(candidates["mape_pct"].median())
        accuracy_n  = _norm01(1.0 / mape_filled)
        safety_score = (0.55 * p_pos_n + 0.45 * accuracy_n)
        candidates["accuracy_score"] = accuracy_n.round(4)
        mape_available = True
    else:
        safety_score = p_pos_n
        candidates["accuracy_score"] = np.nan
        mape_available = False

    if "pred_ann_vol" in candidates.columns and candidates["pred_ann_vol"].notna().any():
        vol_filled = candidates["pred_ann_vol"].fillna(candidates["pred_ann_vol"].median())
        sigma_h = vol_filled * np.sqrt(horizon_days / 252.0)
    else:
        sigma_h = pd.Series(0.0, index=candidates.index)
    utility = candidates["growth_pred"] - 0.5 * RISK_AVERSION[risk_level] * (sigma_h ** 2)
    utility_n = _norm01(utility)

    w_util = W_UTILITY[risk_level]
    if is_crisis and risk_level <= 2:
        w_util = min(w_util + 0.10, 0.55)
    w_safe = w_safe * (1.0 - w_util)
    w_ret  = w_ret * (1.0 - w_util)

    candidates["safety_score"]  = safety_score.round(4)
    candidates["return_score"]  = growth_n.round(4)
    candidates["utility_score"] = utility_n.round(4)
    candidates["score"] = (w_safe * safety_score + w_ret * growth_n + w_util * utility_n).round(4)
    candidates["w_safe"]    = round(w_safe, 3)
    candidates["w_return"]  = round(w_ret, 3)
    candidates["w_utility"] = round(w_util, 3)
    candidates["risk_level"] = risk_level

    top = candidates.nlargest(top_n, "score").reset_index(drop=True)

    regime_type = ("bear" if rrs > 0.55 else ("bull" if rrs < 0.20 else "sideways"))
    vol = candidates["pred_ann_vol"] if "pred_ann_vol" in candidates.columns else pd.Series(dtype=float)
    mape_col = candidates["mape_pct"] if "mape_pct" in candidates.columns else pd.Series(dtype=float)
    meta = {
        "risk_level":      risk_level,
        "level_name":      LEVEL_NAMES_V2[risk_level],
        "w_safe":          round(w_safe, 3),
        "w_return":        round(w_ret, 3),
        "w_utility":       round(w_util, 3),
        "risk_aversion":   RISK_AVERSION[risk_level],
        "vol_tier":        VOL_TIER_QUANTILE[risk_level],
        "vol_filtered":    vol_filtered,
        "sharpe_filtered": sharpe_filtered,
        "min_sharpe_used": round(min_sharpe, 3),
        "dd_quantile":     dd_quantile,
        "cash_included":   cash_added,
        "n_candidates":    len(candidates),
        "mean_pred_vol":   round(float(vol.mean()), 4) if len(vol) > 0 else None,
        "mean_mape_pct":   round(float(mape_col.mean()), 4) if mape_available else None,
        "mape_in_safety":  mape_available,
        "regime_type":     regime_type,
        "rrs":             round(rrs, 3),
        "is_crisis":       is_crisis,
    }
    return top, meta


# ── v4: individual model predictions (M1-M4), no ENS as primary signal ────────

_MODELS_V4 = ["M1", "M2", "M3", "M4"]


def build_pred_frame_v2(
    model_results: pd.DataFrame,
    pred_vol_series: pd.Series,
    current_id: Optional[int],
) -> Tuple[pd.DataFrame, int]:
    """Build scoring DataFrame using individual M1-M4 predictions (no ENS primary).

    Signals constructed:
    - growth_pred        : quality-weighted mean of M1-M4 growth predictions
    - p_positive         : quality-weighted mean of M1-M4 precision (hit rate)
    - mape_pct           : quality-weighted mean MAPE
    - sharpe / maxdd     : mean of models that have them (M1, M2)
    - direction_consensus: fraction of models predicting positive return (0-1)
    - model_agreement    : 1 - clipped CV of predictions (1=all agree, 0=diverge)
    - best_model_growth  : highest individual model growth prediction
    - M1_growth .. M4_growth: raw individual predictions exposed for transparency
    """
    mr = model_results.copy()
    mr.index = mr.index.astype(int)

    pf = pd.DataFrame(index=mr.index)
    pf.index.name = "portfolio_id"
    pf["portfolio_id"] = pf.index

    # ── Per-model raw signals ─────────────────────────────────────────────────
    growth_cols, prec_cols, mape_cols = {}, {}, {}
    for m in _MODELS_V4:
        g_col = f"{m}_growth"
        p_col = f"{m}_precision"
        q_col = f"{m}_mape_pct"
        pf[g_col] = mr[g_col] if g_col in mr.columns else np.nan
        growth_cols[m] = pf[g_col]
        prec_cols[m]   = mr[p_col] if p_col in mr.columns else pd.Series(0.5, index=mr.index)
        mape_cols[m]   = mr[q_col] if q_col in mr.columns else pd.Series(np.nan, index=mr.index)

    # ── Quality weight per model: precision / (mape + eps) ───────────────────
    raw_w = {}
    for m in _MODELS_V4:
        prec = prec_cols[m].fillna(0.5)
        mape = mape_cols[m].fillna(mape_cols[m].median() if mape_cols[m].notna().any() else 50.0)
        raw_w[m] = prec / (mape + 1e-6)

    total_w = sum(raw_w[m] for m in _MODELS_V4).replace(0, 1e-9)

    # ── Weighted consensus signals ────────────────────────────────────────────
    pf["growth_pred"] = sum(raw_w[m] * pf[f"{m}_growth"].fillna(0.0) for m in _MODELS_V4) / total_w
    pf["p_positive"]  = sum(raw_w[m] * prec_cols[m].fillna(0.5) for m in _MODELS_V4) / total_w
    pf["mape_pct"]    = sum(raw_w[m] * mape_cols[m].fillna(50.0) for m in _MODELS_V4) / total_w

    # sharpe and maxdd: only M1, M2 have them
    sharpe_vals = [mr[f"{m}_sharpe"] for m in ["M1", "M2"] if f"{m}_sharpe" in mr.columns]
    maxdd_vals  = [mr[f"{m}_maxdd"]  for m in ["M1", "M2"] if f"{m}_maxdd"  in mr.columns]
    pf["sharpe"] = pd.concat(sharpe_vals, axis=1).mean(axis=1) if sharpe_vals else pd.Series(0.0, index=mr.index)
    pf["maxdd"]  = pd.concat(maxdd_vals,  axis=1).mean(axis=1) if maxdd_vals  else pd.Series(0.0, index=mr.index)

    # ── Consensus / agreement signals ─────────────────────────────────────────
    growth_matrix = pd.concat(
        [pf[f"{m}_growth"].fillna(0.0) for m in _MODELS_V4], axis=1
    )
    growth_matrix.columns = _MODELS_V4

    n_active = growth_matrix.notna().sum(axis=1).clip(lower=1)
    pf["direction_consensus"] = (growth_matrix > 0).sum(axis=1) / n_active

    g_mean = growth_matrix.mean(axis=1)
    g_std  = growth_matrix.std(axis=1).fillna(0.0)
    cv     = g_std / (g_mean.abs() + 1e-6)
    pf["model_agreement"] = (1.0 - cv.clip(0.0, 1.0))

    pf["best_model_growth"] = growth_matrix.max(axis=1)

    # ── Volatility and regime ─────────────────────────────────────────────────
    pf["pred_ann_vol"] = pred_vol_series.reindex(pf.index)

    hit_col = f"M1_regime_{current_id}_hit" if current_id is not None else None
    n_col   = f"M1_regime_{current_id}_n"   if current_id is not None else None
    if hit_col and hit_col in mr.columns:
        pf["regime_hit"]  = mr[hit_col].reindex(pf.index)
        regime_n_samples  = int(mr[n_col].mean()) if n_col in mr.columns else 0
    else:
        pf["regime_hit"]  = np.nan
        regime_n_samples  = 0

    return pf, regime_n_samples


def recommend_v4(
    pred_frame: pd.DataFrame,
    risk_level: int,
    regime_context: dict,
    top_n: int = 3,
    horizon_days: int = HORIZON_DAYS_DEFAULT,
    include_cash: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Recommendation using individual M1-M4 signals (no ENS primary).

    safety_score = 0.40*p_pos + 0.35*direction_consensus + 0.25*model_agreement
    return_score = norm(growth_pred)  for Risk 1-3
                 = norm(best_model_growth) for Risk 4-5 (upside potential)
    utility      = growth_pred - 0.5 * lambda * sigma_h^2  (unchanged)
    """
    df = pred_frame.copy()
    cash_added = False
    if include_cash:
        df, cash_added = _add_cash_candidate(df, risk_level, regime_context, horizon_days)
    df["is_cash"] = (df.index.astype(int) == CASH_PORTFOLIO_ID)

    rrs       = float(regime_context.get("regime_risk_score", 0.35))
    is_crisis = bool(regime_context.get("is_crisis") or regime_context.get("is_bear_outlook"))

    # ── Regime weight adjustment (identical to v3) ───────────────────────────
    w_safe_base, _ = RISK_V2_WEIGHTS[risk_level]
    if is_crisis:
        w_safe = min(w_safe_base + (0.25 if risk_level <= 2 else 0.15), 0.95)
    elif rrs > 0.55:
        w_safe = min(w_safe_base + 0.20, 0.95)
    elif rrs < 0.20:
        w_safe = max(w_safe_base - 0.10, 0.05)
    else:
        w_safe = w_safe_base
    w_ret = 1.0 - w_safe

    # ── Vol-tier filter (identical to v3) ────────────────────────────────────
    vol_filtered = False
    if "pred_ann_vol" in df.columns and df["pred_ann_vol"].notna().sum() >= top_n:
        q_lo, q_hi = VOL_TIER_QUANTILE[risk_level]
        if is_crisis and risk_level <= 2:
            q_hi = max(q_lo + 0.10, q_hi - 0.10)
        vol_lo = df["pred_ann_vol"].quantile(q_lo) if q_lo > 0.0 else -np.inf
        vol_hi = df["pred_ann_vol"].quantile(q_hi) if q_hi < 1.0 else np.inf
        tier   = df[(df["pred_ann_vol"] >= vol_lo) & (df["pred_ann_vol"] <= vol_hi)].copy()
        if len(tier) >= top_n:
            candidates = tier
            vol_filtered = True
        else:
            candidates = df.copy()
    else:
        candidates = df.copy()

    # ── Sharpe + MaxDD filters (identical to v3) ─────────────────────────────
    sharpe_filtered = False
    min_sharpe = 0.25
    if is_crisis:
        min_sharpe = 0.55 if risk_level <= 2 else 0.40
    elif rrs > 0.55:
        min_sharpe = 0.45 if risk_level <= 2 else 0.35
    if "sharpe" in candidates.columns and candidates["sharpe"].notna().sum() >= top_n:
        _f = candidates[candidates["sharpe"] >= min_sharpe]
        if len(_f) >= top_n:
            candidates = _f
            sharpe_filtered = True

    dd_quantile = None
    if "maxdd" in candidates.columns and candidates["maxdd"].notna().sum() >= top_n:
        base_q = 0.60 if risk_level == 1 else (0.50 if risk_level == 2 else 0.40)
        if is_crisis:
            base_q = min(base_q + 0.10, 0.90)
        thr_dd = candidates["maxdd"].quantile(base_q)
        _f = candidates[candidates["maxdd"] >= thr_dd]
        if len(_f) >= top_n:
            candidates = _f
            dd_quantile = base_q

    # ── Scoring ───────────────────────────────────────────────────────────────
    candidates = candidates.copy()
    p_pos_n = _norm01(candidates["p_positive"])

    # Consensus signals
    has_consensus = ("direction_consensus" in candidates.columns
                     and candidates["direction_consensus"].notna().sum() >= top_n)
    has_agreement = ("model_agreement" in candidates.columns
                     and candidates["model_agreement"].notna().sum() >= top_n)

    consensus_n = _norm01(candidates["direction_consensus"]) if has_consensus else p_pos_n
    agreement_n = _norm01(candidates["model_agreement"])     if has_agreement else pd.Series(0.5, index=candidates.index)

    # safety_score: weighted precision + directional consensus + magnitude agreement
    safety_score = 0.40 * p_pos_n + 0.35 * consensus_n + 0.25 * agreement_n

    # return_score: Risk 1-3 = quality-weighted mean; Risk 4-5 = best single model upside
    if risk_level >= 4 and "best_model_growth" in candidates.columns:
        return_score = _norm01(candidates["best_model_growth"])
    else:
        return_score = _norm01(candidates["growth_pred"])

    # utility (risk-aversion penalizes vol)
    if "pred_ann_vol" in candidates.columns and candidates["pred_ann_vol"].notna().any():
        vol_filled = candidates["pred_ann_vol"].fillna(candidates["pred_ann_vol"].median())
        sigma_h = vol_filled * np.sqrt(horizon_days / 252.0)
    else:
        sigma_h = pd.Series(0.0, index=candidates.index)
    utility   = candidates["growth_pred"] - 0.5 * RISK_AVERSION[risk_level] * (sigma_h ** 2)
    utility_n = _norm01(utility)

    w_util  = W_UTILITY[risk_level]
    if is_crisis and risk_level <= 2:
        w_util = min(w_util + 0.10, 0.55)
    w_safe  = w_safe * (1.0 - w_util)
    w_ret   = w_ret  * (1.0 - w_util)

    candidates["safety_score"]  = safety_score.round(4)
    candidates["return_score"]  = return_score.round(4)
    candidates["utility_score"] = utility_n.round(4)
    candidates["score"] = (w_safe * safety_score + w_ret * return_score + w_util * utility_n).round(4)
    candidates["w_safe"]    = round(w_safe, 3)
    candidates["w_return"]  = round(w_ret, 3)
    candidates["w_utility"] = round(w_util, 3)
    candidates["risk_level"] = risk_level

    top = candidates.nlargest(top_n, "score").reset_index(drop=True)

    regime_type = "bear" if rrs > 0.55 else ("bull" if rrs < 0.20 else "sideways")
    vol     = candidates.get("pred_ann_vol", pd.Series(dtype=float))
    mape_c  = candidates.get("mape_pct",     pd.Series(dtype=float))
    meta = {
        "risk_level":         risk_level,
        "level_name":         LEVEL_NAMES_V2[risk_level],
        "w_safe":             round(w_safe, 3),
        "w_return":           round(w_ret, 3),
        "w_utility":          round(w_util, 3),
        "risk_aversion":      RISK_AVERSION[risk_level],
        "vol_tier":           VOL_TIER_QUANTILE[risk_level],
        "vol_filtered":       vol_filtered,
        "sharpe_filtered":    sharpe_filtered,
        "min_sharpe_used":    round(min_sharpe, 3),
        "dd_quantile":        dd_quantile,
        "cash_included":      cash_added,
        "n_candidates":       len(candidates),
        "mean_pred_vol":      round(float(vol.mean()), 4) if vol.notna().any() else None,
        "mean_mape_pct":      round(float(mape_c.mean()), 4) if mape_c.notna().any() else None,
        "regime_type":        regime_type,
        "rrs":                round(rrs, 3),
        "is_crisis":          is_crisis,
        "return_score_basis": "best_model_growth" if risk_level >= 4 else "quality_weighted_mean",
    }
    return top, meta


# ─────────────────────────────────────────────────────────────────────────────
# v5 — Hybrid: ENS primary + individual M1-M4 consensus filter
# ─────────────────────────────────────────────────────────────────────────────

def build_pred_frame_v3(
    model_results: pd.DataFrame,
    pred_vol_series: pd.Series,
    current_id,
) -> Tuple[pd.DataFrame, int]:
    """Build pred_frame using ENS as primary signal + M1-M4 individual signals.

    Primary (from ENS): growth_pred, p_positive, mape_pct, sharpe, maxdd.
    Supplementary (from M1-M4): direction_consensus, model_agreement, best_model_growth.
    Falls back to quality-weighted individual means when ENS columns are absent.
    """
    mr = model_results.copy()
    if "portfolio_id" in mr.columns:
        mr = mr.set_index("portfolio_id")
    pf = pd.DataFrame(index=mr.index)

    # ── Individual M1-M4 signals (always computed) ───────────────────────────
    growth_cols_data, prec_cols_data, mape_cols_data = {}, {}, {}
    for m in _MODELS_V4:
        g_col = f"{m}_growth"
        p_col = f"{m}_precision"
        q_col = f"{m}_mape_pct"
        pf[g_col] = mr[g_col] if g_col in mr.columns else np.nan
        growth_cols_data[m] = pf[g_col]
        prec_cols_data[m]   = mr[p_col] if p_col in mr.columns else pd.Series(0.5, index=mr.index)
        mape_cols_data[m]   = mr[q_col] if q_col in mr.columns else pd.Series(np.nan, index=mr.index)

    growth_matrix = pd.concat(
        [pf[f"{m}_growth"] for m in _MODELS_V4], axis=1
    )
    growth_matrix.columns = _MODELS_V4
    n_active = growth_matrix.notna().sum(axis=1).clip(lower=1)

    pf["direction_consensus"] = (growth_matrix.fillna(0.0) > 0).sum(axis=1) / n_active
    pf.loc[n_active == 0, "direction_consensus"] = np.nan

    g_mean = growth_matrix.mean(axis=1)
    g_std  = growth_matrix.std(axis=1).fillna(0.0)
    cv     = g_std / (g_mean.abs() + 1e-6)
    pf["model_agreement"] = (1.0 - cv.clip(0.0, 1.0))
    pf.loc[growth_matrix.notna().sum(axis=1) < 2, "model_agreement"] = np.nan

    pf["best_model_growth"] = growth_matrix.max(axis=1)

    # ── Primary signal: ENS if available, else quality-weighted individual ───
    ens_growth = mr.get("ENS_growth",    pd.Series(dtype=float))
    ens_prec   = mr.get("ENS_precision", pd.Series(dtype=float))
    ens_mape   = mr.get("ENS_mape_pct",  pd.Series(dtype=float))
    ens_sharpe = mr.get("ENS_sharpe",    pd.Series(dtype=float))
    ens_maxdd  = mr.get("ENS_maxdd",     pd.Series(dtype=float))

    has_ens = isinstance(ens_growth, pd.Series) and ens_growth.notna().sum() >= 3

    if has_ens:
        # Precision³-weighted ensemble: amplifies quality differences (M1 ElasticNet ~35% vs uniform ~26%)
        _avail = [m for m in _MODELS_V4
                  if f"{m}_growth" in mr.columns and f"{m}_precision" in mr.columns]
        if len(_avail) >= 2:
            prec_cu = pd.concat(
                [prec_cols_data[m].reindex(pf.index).fillna(0.5) ** 3 for m in _avail], axis=1
            )
            prec_cu.columns = _avail
            total_q = prec_cu.sum(axis=1).clip(lower=1e-9)
            growth_m = pd.concat(
                [pf[f"{m}_growth"].fillna(0.0) for m in _avail], axis=1
            )
            growth_m.columns = _avail
            pf["growth_pred"] = (prec_cu * growth_m).sum(axis=1) / total_q
            prec_sq = prec_cu ** (2.0 / 3.0)
            pf["p_positive"] = (prec_sq * prec_sq).sum(axis=1) / prec_sq.sum(axis=1).clip(lower=1e-9)
        else:
            pf["growth_pred"] = ens_growth.reindex(pf.index)
            pf["p_positive"]  = ens_prec.reindex(pf.index)
        pf["mape_pct"]    = ens_mape.reindex(pf.index) if isinstance(ens_mape, pd.Series) else np.nan
        pf["sharpe"] = (ens_sharpe.reindex(pf.index)
                        if isinstance(ens_sharpe, pd.Series) and ens_sharpe.notna().any()
                        else pd.Series(np.nan, index=pf.index))
        pf["maxdd"] = (ens_maxdd.reindex(pf.index)
                       if isinstance(ens_maxdd, pd.Series) and ens_maxdd.notna().any()
                       else pd.Series(np.nan, index=pf.index))
    else:
        raw_w = {}
        for m in _MODELS_V4:
            prec = prec_cols_data[m].fillna(0.5)
            mape = mape_cols_data[m].fillna(
                mape_cols_data[m].median() if mape_cols_data[m].notna().any() else 50.0
            )
            raw_w[m] = prec / (mape + 1e-6)
        total_w = sum(raw_w[m] for m in _MODELS_V4).replace(0, 1e-9)
        pf["growth_pred"] = sum(raw_w[m] * pf[f"{m}_growth"].fillna(0.0) for m in _MODELS_V4) / total_w
        pf["p_positive"]  = sum(raw_w[m] * prec_cols_data[m].fillna(0.5) for m in _MODELS_V4) / total_w
        pf["mape_pct"]    = sum(raw_w[m] * mape_cols_data[m].fillna(50.0) for m in _MODELS_V4) / total_w
        sharpe_vals = [mr[f"{m}_sharpe"] for m in ["M1", "M2"] if f"{m}_sharpe" in mr.columns]
        maxdd_vals  = [mr[f"{m}_maxdd"]  for m in ["M1", "M2"] if f"{m}_maxdd"  in mr.columns]
        pf["sharpe"] = pd.concat(sharpe_vals, axis=1).mean(axis=1) if sharpe_vals else pd.Series(0.0, index=mr.index)
        pf["maxdd"]  = pd.concat(maxdd_vals,  axis=1).mean(axis=1) if maxdd_vals  else pd.Series(0.0, index=mr.index)

    pf["pred_ann_vol"] = pred_vol_series.reindex(pf.index)

    # Fallback: if fewer than 10 portfolios have vol data (index mismatch),
    # load historical annual volatility from portfolios_features.parquet
    if pf["pred_ann_vol"].notna().sum() < 10:
        feat_candidates = [
            Path(__file__).parent.parent / "data" / "processed" / "portfolios_features.parquet",
            Path("data") / "processed" / "portfolios_features.parquet",
        ]
        for feat_path in feat_candidates:
            if feat_path.exists():
                feat = pd.read_parquet(feat_path)
                if "portfolio_id" in feat.columns and "annual_volatility" in feat.columns:
                    hist_vol = feat.set_index("portfolio_id")["annual_volatility"]
                    merged = hist_vol.reindex(pf.index)
                    if merged.notna().sum() >= 10:
                        pf["pred_ann_vol"] = merged
                        print(f"Vol (historical fallback): {merged.notna().sum()} portfolios, "
                              f"range=[{merged.min():.3f}, {merged.max():.3f}]")
                        break

    hit_col = f"ENS_regime_{current_id}_hit" if current_id is not None else None
    n_col   = f"ENS_regime_{current_id}_n"   if current_id is not None else None
    if hit_col and hit_col in mr.columns:
        pf["regime_hit"] = mr[hit_col].reindex(pf.index)
        regime_n_samples = int(mr[n_col].mean()) if n_col in mr.columns else 0
    else:
        hit_col_m1 = f"M1_regime_{current_id}_hit" if current_id is not None else None
        n_col_m1   = f"M1_regime_{current_id}_n"   if current_id is not None else None
        if hit_col_m1 and hit_col_m1 in mr.columns:
            pf["regime_hit"] = mr[hit_col_m1].reindex(pf.index)
            regime_n_samples = int(mr[n_col_m1].mean()) if n_col_m1 in mr.columns else 0
        else:
            pf["regime_hit"] = np.nan
            regime_n_samples = 0

    pf["portfolio_id"] = pf.index.astype(int)
    pf["_is_ens"] = has_ens
    return pf, regime_n_samples


MIN_MODEL_AGREEMENT: dict = {1: 0.15, 2: 0.10, 3: 0.0, 4: 0.0, 5: 0.0}


def recommend_v5(
    pred_frame: pd.DataFrame,
    risk_level: int,
    regime_context: dict,
    top_n: int = 3,
    horizon_days: int = HORIZON_DAYS_DEFAULT,
    include_cash: bool = True,
    exclude_portfolio_ids: Optional[set] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Hybrid: ENS primary signal + individual M1-M4 model-agreement filter.

    safety_score = 0.55 * p_pos_n + 0.45 * agreement_n
    return_score = norm(growth_pred)       for Risk 1-3
                 = norm(best_model_growth) for Risk 4-5 (upside)
    filter:
        Risk 1: model_agreement >= 0.15 (all models must roughly agree)
        Risk 2: model_agreement >= 0.10
        Risk 3-5: no consensus filter
    """
    df = pred_frame.copy()
    cash_added = False
    if include_cash:
        df, cash_added = _add_cash_candidate(df, risk_level, regime_context, horizon_days)
    df["is_cash"] = (df.index.astype(int) == CASH_PORTFOLIO_ID)

    if exclude_portfolio_ids:
        excluded = {int(pid) for pid in exclude_portfolio_ids}
        if excluded:
            df = df[~df.index.astype(int).isin(excluded)].copy()

    rrs       = float(regime_context.get("regime_risk_score", 0.35))
    is_crisis = bool(regime_context.get("is_crisis") or regime_context.get("is_bear_outlook"))

    w_safe_base, _ = RISK_V2_WEIGHTS[risk_level]
    if is_crisis:
        w_safe = min(w_safe_base + (0.25 if risk_level <= 2 else 0.15), 0.95)
    elif rrs > 0.55:
        w_safe = min(w_safe_base + 0.20, 0.95)
    elif rrs < 0.20:
        w_safe = max(w_safe_base - 0.10, 0.05)
    else:
        w_safe = w_safe_base
    w_ret = 1.0 - w_safe

    # ── Vol-tier filter ──────────────────────────────────────────────────────
    vol_filtered = False
    candidates = df.copy()
    if "pred_ann_vol" in df.columns and df["pred_ann_vol"].notna().sum() >= top_n:
        q_lo, q_hi = VOL_TIER_QUANTILE[risk_level]
        if is_crisis and risk_level <= 2:
            q_hi = max(q_lo + 0.10, q_hi - 0.10)
        vol_lo = df["pred_ann_vol"].quantile(q_lo) if q_lo > 0.0 else -np.inf
        vol_hi = df["pred_ann_vol"].quantile(q_hi) if q_hi < 1.0 else np.inf
        tier   = df[(df["pred_ann_vol"] >= vol_lo) & (df["pred_ann_vol"] <= vol_hi)].copy()
        if len(tier) >= top_n:
            candidates = tier
            vol_filtered = True

    # ── Sharpe + MaxDD filters ───────────────────────────────────────────────
    sharpe_filtered = False
    min_sharpe = 0.25
    if is_crisis:
        min_sharpe = 0.55 if risk_level <= 2 else 0.40
    elif rrs > 0.55:
        min_sharpe = 0.45 if risk_level <= 2 else 0.35
    if "sharpe" in candidates.columns and candidates["sharpe"].notna().sum() >= top_n:
        _f = candidates[candidates["sharpe"] >= min_sharpe]
        if len(_f) >= top_n:
            candidates = _f
            sharpe_filtered = True

    if "maxdd" in candidates.columns and candidates["maxdd"].notna().sum() >= top_n:
        base_q = 0.60 if risk_level == 1 else (0.50 if risk_level == 2 else 0.40)
        if is_crisis:
            base_q = min(base_q + 0.10, 0.90)
        thr_dd = candidates["maxdd"].quantile(base_q)
        _f = candidates[candidates["maxdd"] >= thr_dd]
        if len(_f) >= top_n:
            candidates = _f

    # ── Consensus filter for conservative profiles ───────────────────────────
    agreement_filtered = False
    min_agree = MIN_MODEL_AGREEMENT.get(risk_level, 0.0)
    if min_agree > 0.0 and "model_agreement" in candidates.columns:
        agree_vals = candidates["model_agreement"].fillna(0.0)
        _f = candidates[agree_vals >= min_agree]
        if len(_f) >= top_n:
            candidates = _f
            agreement_filtered = True

    # ── Scoring ───────────────────────────────────────────────────────────────
    candidates = candidates.copy()
    p_pos_n = _norm01(candidates["p_positive"])
    agr_col = (candidates["model_agreement"].fillna(0.0)
               if "model_agreement" in candidates.columns
               else pd.Series(0.5, index=candidates.index))
    agreement_n  = _norm01(agr_col)
    safety_score = 0.55 * p_pos_n + 0.45 * agreement_n

    has_ens = bool(candidates.get("_is_ens", pd.Series(False)).any())
    if risk_level >= 4 and "best_model_growth" in candidates.columns:
        return_score = _norm01(candidates["best_model_growth"])
        return_basis = "best_model_growth"
    else:
        return_score = _norm01(candidates["growth_pred"])
        return_basis = "ens_growth" if has_ens else "quality_weighted_mean"

    if "pred_ann_vol" in candidates.columns and candidates["pred_ann_vol"].notna().any():
        vol_filled = candidates["pred_ann_vol"].fillna(candidates["pred_ann_vol"].median())
        sigma_h = vol_filled * np.sqrt(horizon_days / 252.0)
    else:
        sigma_h = pd.Series(0.0, index=candidates.index)
    utility   = candidates["growth_pred"] - 0.5 * RISK_AVERSION[risk_level] * (sigma_h ** 2)
    utility_n = _norm01(utility)

    w_util = W_UTILITY[risk_level]
    if risk_level >= 4:
        w_safe = max(w_safe - 0.10, 0.01)
        w_ret  = min(w_ret + 0.10, 0.99)
        w_util = max(w_util - 0.05, 0.0)
    if is_crisis and risk_level <= 2:
        w_util = min(w_util + 0.10, 0.55)
    w_safe = w_safe * (1.0 - w_util)
    w_ret  = w_ret  * (1.0 - w_util)

    candidates["safety_score"]  = safety_score.round(4)
    candidates["return_score"]  = return_score.round(4)
    candidates["utility_score"] = utility_n.round(4)
    candidates["score"] = (w_safe * safety_score + w_ret * return_score + w_util * utility_n).round(4)
    candidates["w_safe"]    = round(w_safe, 3)
    candidates["w_return"]  = round(w_ret, 3)
    candidates["w_utility"] = round(w_util, 3)
    candidates["risk_level"] = risk_level

    top = candidates.nlargest(top_n, "score").reset_index(drop=True)

    regime_type = "bear" if rrs > 0.55 else ("bull" if rrs < 0.20 else "sideways")
    vol_c  = candidates.get("pred_ann_vol", pd.Series(dtype=float))
    mape_c = candidates.get("mape_pct",     pd.Series(dtype=float))
    meta = {
        "risk_level":          risk_level,
        "level_name":          LEVEL_NAMES_V2[risk_level],
        "w_safe":              round(w_safe, 3),
        "w_return":            round(w_ret, 3),
        "w_utility":           round(w_util, 3),
        "risk_aversion":       RISK_AVERSION[risk_level],
        "vol_tier":            VOL_TIER_QUANTILE[risk_level],
        "vol_filtered":        vol_filtered,
        "sharpe_filtered":     sharpe_filtered,
        "agreement_filtered":  agreement_filtered,
        "min_agreement_used":  min_agree,
        "n_candidates":        len(candidates),
        "cash_included":       cash_added,
        "mean_pred_vol":       round(float(vol_c.mean()),  4) if vol_c.notna().any()  else None,
        "mean_mape_pct":       round(float(mape_c.mean()), 4) if mape_c.notna().any() else None,
        "regime_type":         regime_type,
        "rrs":                 round(rrs, 3),
        "is_crisis":           is_crisis,
        "return_score_basis":  return_basis,
    }
    return top, meta
