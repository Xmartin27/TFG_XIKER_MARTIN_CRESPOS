"""
nb03_hmm.py — HMM regime detection functions for Notebook 03.

All computation extracted from run_nb03.py so the notebook can call
clean functions and display results with interpretations inline.
"""
import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sys
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
TRADING_DAYS = 252


# ── 1. Data preparation ───────────────────────────────────────────────────────

def build_market_proxy(returns: pd.DataFrame, macro: pd.DataFrame) -> pd.Series:
    """Return daily log-return series for the market proxy.

    Uses S&P 500 from macro if available, otherwise equal-weighted universe.
    """
    if "S&P 500" in macro.columns:
        sp500_px = macro["S&P 500"].dropna()
        mkt_ret  = np.log(sp500_px / sp500_px.shift(1)).dropna()
        mkt_ret.name = "market_return"
        print(f"Market proxy: S&P 500 ({len(mkt_ret)} days)")
    else:
        mkt_ret = returns.mean(axis=1).rename("market_return")
        print("Market proxy: equal-weighted universe")
    return mkt_ret


def build_hmm_features(
    mkt_ret: pd.Series,
    macro: pd.DataFrame,
    returns_index: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """Build and scale the HMM observable feature matrix.

    Features: market_return, realized_vol_21d (rv²), realized_vol_63d (rv²), vix (optional).

    Returns:
        features   — raw DataFrame indexed by date
        X_scaled   — standardized numpy array
        scaler     — fitted StandardScaler
    """
    vix = macro["VIX"].dropna() if "VIX" in macro.columns else None

    aligned = mkt_ret.reindex(returns_index).fillna(0.0)
    rv21 = (aligned ** 2).rolling(21).mean().rename("realized_vol_21d")
    rv63 = (aligned ** 2).rolling(63).mean().rename("realized_vol_63d")

    parts = [aligned, rv21, rv63]
    if vix is not None:
        parts.append(vix.reindex(returns_index).ffill().rename("vix"))

    features = pd.concat(parts, axis=1).dropna()
    print(f"HMM feature matrix: {features.shape}  cols={list(features.columns)}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)
    return features, X_scaled, scaler


# ── 2. Model selection ────────────────────────────────────────────────────────

def fit_hmm_bic(
    X_scaled: np.ndarray,
    state_range: Tuple[int, int] = (2, 6),
    n_inits: int = 10,
) -> Tuple[GaussianHMM, pd.DataFrame, int]:
    """Train GaussianHMMs with varying state counts and select by BIC.

    Returns:
        best_model  — HMM with lowest BIC
        results_df  — DataFrame with n_states, bic, aic, log_likelihood per run
        optimal_n   — number of states in best_model
    """
    print("\nSelecting optimal number of HMM states...")
    records   = []
    best_model, best_bic = None, np.inf

    for n in range(state_range[0], state_range[1] + 1):
        models = []
        for seed in range(n_inits):
            m = GaussianHMM(n_components=n, covariance_type="full",
                            n_iter=300, tol=1e-4, random_state=seed)
            try:
                m.fit(X_scaled)
                ll  = m.score(X_scaled)
                k   = n**2 + n * X_scaled.shape[1] + n * X_scaled.shape[1]**2
                bic = -2 * ll * len(X_scaled) + k * np.log(len(X_scaled))
                aic = -2 * ll * len(X_scaled) + 2 * k
                models.append((bic, m, ll, aic))
            except Exception:
                pass

        if models:
            best_run = min(models, key=lambda x: x[0])
            records.append({"n_states": n, "bic": best_run[0], "aic": best_run[3],
                            "log_likelihood": best_run[2], "model": best_run[1]})
            print(f"  n={n}: BIC={best_run[0]:.1f}  AIC={best_run[3]:.1f}")
            if best_run[0] < best_bic:
                best_bic   = best_run[0]
                best_model = best_run[1]

    results_df = pd.DataFrame(records)
    optimal_n  = int(results_df.loc[results_df["bic"].idxmin(), "n_states"])
    best_model = results_df.loc[results_df["bic"].idxmin(), "model"]
    print(f"Optimal states (BIC): {optimal_n}")
    return best_model, results_df, optimal_n


# ── 3. State labeling ─────────────────────────────────────────────────────────

def semantic_label(state_id: int, model: GaussianHMM, feat_idx: Dict[str, int]) -> str:
    """Assign a rich semantic label to an HMM state based on its mean vector.

    Thresholds are in standardized space (mean=0, std=1).
    """
    m = model.means_[state_id]
    ret_val = m[feat_idx.get("market_return", 0)]
    vol_val = m[feat_idx.get("realized_vol_21d", 1)]
    vix_val = m[feat_idx.get("vix", 3)] if "vix" in feat_idx else 0.0

    high_vol = vol_val >  0.5
    low_vol  = vol_val < -0.3
    bull     = ret_val >  0.3
    bear     = ret_val < -0.3
    high_vix = vix_val >  0.5

    if bear and high_vol and high_vix:   return "Crisis / Crash"
    elif bear and high_vol:              return "Bear High-Vol"
    elif bear and not high_vol:          return "Bear Low-Vol"
    elif bull and low_vol:               return "Bull Low-Vol"
    elif bull and high_vol:              return "Bull High-Vol"
    elif bull:                           return "Bull Stable"
    elif high_vol and high_vix:          return "High Uncertainty"
    elif low_vol and not bull and not bear: return "Sideways Low-Vol"
    else:                                return "Transition"


def compute_state_labels(model: GaussianHMM, feat_idx: Dict[str, int]) -> Dict[int, str]:
    """Return {state_id: semantic_label} for all states in the model."""
    return {s: semantic_label(s, model, feat_idx) for s in range(model.n_components)}


# ── 4. Regime features ────────────────────────────────────────────────────────

def compute_regime_features(
    model: GaussianHMM,
    X_scaled: np.ndarray,
    features: pd.DataFrame,
    state_labels: Dict[int, str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate the full regime feature DataFrame for downstream notebooks.

    Columns produced:
        current_regime, current_regime_id, prob_transition,
        predicted_prob_bull_22d, predicted_prob_bear_22d,
        predicted_prob_transition_22d, regime_stability, regime_duration,
        prob_state_0 … prob_state_N

    Compatibility:
        Legacy 21d columns are also emitted as aliases of the 22d projections.

    Returns:
        regime_features — DataFrame indexed by date
        states          — integer array of decoded states
        probs           — (T × n_states) posterior probability matrix
    """
    n_states = model.n_components
    states   = model.predict(X_scaled)
    probs    = model.predict_proba(X_scaled)

    state_probs_df    = pd.DataFrame(probs, index=features.index,
                                     columns=[f"prob_state_{s}" for s in range(n_states)])
    current_regime_id = pd.Series(states, index=features.index, name="current_regime_id")
    current_regime    = pd.Series([state_labels[s] for s in states],
                                   index=features.index, name="current_regime")

    persist_probs   = np.array([model.transmat_[s, s] for s in states])
    prob_transition = pd.Series(1 - persist_probs, index=features.index, name="prob_transition")

    transmat_22  = np.linalg.matrix_power(model.transmat_, 22)
    forward_probs = probs @ transmat_22

    bull_states  = [s for s, lbl in state_labels.items() if "Bull"   in lbl]
    bear_states  = [s for s, lbl in state_labels.items() if "Bear"   in lbl or "Crisis" in lbl]
    trans_states = [s for s in range(n_states) if s not in bull_states and s not in bear_states]

    def _fwd(idxs):
        if idxs:
            return forward_probs[:, idxs].sum(axis=1)
        return np.zeros(len(features))

    prob_bull_22  = pd.Series(_fwd(bull_states),  index=features.index, name="predicted_prob_bull_22d")
    prob_bear_22  = pd.Series(_fwd(bear_states),  index=features.index, name="predicted_prob_bear_22d")
    prob_trans_22 = pd.Series(_fwd(trans_states), index=features.index, name="predicted_prob_transition_22d")

    prob_bull_21  = prob_bull_22.rename("predicted_prob_bull_21d")
    prob_bear_21  = prob_bear_22.rename("predicted_prob_bear_21d")
    prob_trans_21 = prob_trans_22.rename("predicted_prob_transition_21d")

    durations = [1]
    dur = 1
    for i in range(1, len(states)):
        dur = dur + 1 if states[i] == states[i - 1] else 1
        durations.append(dur)
    regime_duration  = pd.Series(durations,  index=features.index, name="regime_duration")
    regime_stability = pd.Series([model.transmat_[s, s] for s in states],
                                  index=features.index, name="regime_stability")

    regime_features = pd.concat([
        current_regime, current_regime_id, prob_transition,
        prob_bull_22, prob_bear_22, prob_trans_22,
        prob_bull_21, prob_bear_21, prob_trans_21,
        regime_stability, regime_duration, state_probs_df,
    ], axis=1)
    print(f"\nRegime features: {regime_features.shape}  cols={list(regime_features.columns)}")
    return regime_features, states, probs


# ── 5. State statistics ───────────────────────────────────────────────────────

def compute_state_stats(
    model: GaussianHMM,
    states: np.ndarray,
    features: pd.DataFrame,
    state_labels: Dict[int, str],
    trading_days: int = 252,
) -> pd.DataFrame:
    """Build a summary table of per-state market statistics."""
    rows = []
    for s in range(model.n_components):
        mask   = states == s
        s_rets = features.loc[features.index[mask], "market_return"]
        ann_ret = s_rets.mean() * trading_days * 100
        ann_vol = s_rets.std() * np.sqrt(trading_days) * 100
        rows.append({
            "State":       s,
            "Label":       state_labels[s],
            "Count":       int(mask.sum()),
            "Pct_Days":    f"{mask.mean()*100:.1f}%",
            "AnnReturn%":  round(ann_ret, 2),
            "AnnVol%":     round(ann_vol, 2),
            "Sharpe":      round(s_rets.mean() / (s_rets.std() + 1e-9), 4),
            "Persistence": round(float(model.transmat_[s, s]), 4),
            "Min_Return":  round(float(s_rets.min()), 5),
            "Max_Return":  round(float(s_rets.max()), 5),
        })
    df = pd.DataFrame(rows).set_index("State")
    print("\nState statistics:")
    print(df.to_string())
    return df


# ── 6. Visualizations ─────────────────────────────────────────────────────────

def plot_regimes(
    features: pd.DataFrame,
    states: np.ndarray,
    state_labels: Dict[int, str],
    probs: np.ndarray,
    fig_dir: Path,
) -> None:
    """3-panel regime plot: returns by state, state probabilities, cumulative return."""
    n_states  = len(state_labels)
    PALETTE   = plt.cm.tab10.colors
    color_map = {s: PALETTE[s % 10] for s in range(n_states)}

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    ax = axes[0]
    for s in range(n_states):
        mask = states == s
        ax.scatter(features.index[mask], features["market_return"].values[mask] * 100,
                   c=[color_map[s]], s=2, alpha=0.5, label=f"{s}: {state_labels[s]}")
    ax.set_ylabel("Daily Return (%)"); ax.set_title("Market Returns by HMM Regime")
    ax.legend(markerscale=5, fontsize=7, ncol=3); ax.grid(alpha=0.3)

    ax = axes[1]
    for s in range(n_states):
        ax.fill_between(features.index, probs[:, s], alpha=0.4,
                        color=color_map[s], label=f"{s}: {state_labels[s]}")
    ax.set_ylabel("Probability"); ax.set_title("State Probabilities Over Time")
    ax.legend(fontsize=7, ncol=3); ax.set_ylim(0, 1); ax.grid(alpha=0.3)

    ax = axes[2]
    cum_ret = (1 + features["market_return"]).cumprod()
    ax.plot(features.index, cum_ret, color="#1f77b4", lw=1)
    for s in range(n_states):
        lbl = state_labels[s]
        if "Crisis" in lbl or "Bear" in lbl:
            mask    = states == s
            changes = np.diff(mask.astype(int))
            starts  = list(np.where(changes == 1)[0] + 1)
            ends    = list(np.where(changes == -1)[0] + 1)
            if mask[0]:  starts = [0] + starts
            if mask[-1]: ends   = ends + [len(mask)]
            for st, en in zip(starts, ends):
                ax.axvspan(features.index[st],
                           features.index[min(en, len(features.index) - 1)],
                           alpha=0.15, color="#d62728")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Cumulative Market Return (bear/crisis states shaded)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = fig_dir / "03_hmm_regimes_v5.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved: {out}")


def plot_transition_heatmap(
    model: GaussianHMM,
    state_labels: Dict[int, str],
    fig_dir: Path,
) -> None:
    """Seaborn heatmap of the HMM transition matrix."""
    try:
        import seaborn as sns
    except ImportError:
        print("seaborn not available — transition heatmap skipped")
        return

    n = model.n_components
    trans = pd.DataFrame(
        model.transmat_,
        index=[f"{s}:{state_labels[s][:12]}" for s in range(n)],
        columns=[f"{s}:{state_labels[s][:12]}" for s in range(n)],
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(trans, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("HMM Transition Matrix")
    plt.tight_layout()
    out = fig_dir / "03_hmm_transition_v5.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved: {out}")


# ── 7. Optional DAG learning ──────────────────────────────────────────────────

def learn_dag(
    features: pd.DataFrame,
    states: np.ndarray,
    state_labels: Dict[int, str],
    processed_dir: Path,
) -> dict:
    """Hill-Climb BIC structure learning (requires pgmpy). Returns dag_summary dict."""
    try:
        from pgmpy.estimators import HillClimbSearch, BicScore
    except ImportError:
        print("pgmpy not available — DAG learning skipped")
        return {}

    disc = features[["market_return", "realized_vol_21d", "realized_vol_63d"]].copy()
    if "vix" in features.columns:
        disc["vix"] = features["vix"]
    for col in disc.columns:
        disc[col] = pd.qcut(disc[col], q=4, labels=["L", "M", "H", "VH"], duplicates="drop")
    disc = disc.dropna()
    disc["regime"] = pd.Categorical([state_labels[s] for s in states[features.index.isin(disc.index)]])

    try:
        hc  = HillClimbSearch(disc)
        dag = hc.estimate(scoring_method=BicScore(disc), max_indegree=3, max_iter=500)
        edges = list(dag.edges())
        print(f"DAG edges (HC): {edges}")
        summary = {"method": "HillClimb-BIC", "edges": edges,
                   "n_nodes": len(disc.columns), "n_edges": len(edges)}
    except Exception as e:
        print(f"DAG learning failed: {e}")
        summary = {"method": "failed", "edges": [], "error": str(e)}

    out = processed_dir / "hmm_dag_structures.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {out}")
    return summary


# ── 8. Save outputs ───────────────────────────────────────────────────────────

def save_all_outputs(
    model: GaussianHMM,
    scaler: StandardScaler,
    state_labels: Dict[int, str],
    features: pd.DataFrame,
    X_scaled: np.ndarray,
    regime_features: pd.DataFrame,
    state_stats: pd.DataFrame,
    processed_dir: Path,
) -> None:
    """Persist all N03 outputs used by downstream notebooks."""
    n = model.n_components
    states = model.predict(X_scaled)
    probs  = model.predict_proba(X_scaled)

    joblib.dump({"model": model, "scaler": scaler, "state_labels": state_labels,
                 "feature_names": list(features.columns)},
                processed_dir / "hmm_model.pkl")

    regime_features.to_parquet(processed_dir / "hmm_regime_features.parquet")
    np.save(processed_dir / "hmm_transmat.npy", model.transmat_)
    state_stats.to_parquet(processed_dir / "hmm_state_stats.parquet")

    trans_df = pd.DataFrame(model.transmat_,
        index=[f"from_{s}" for s in range(n)],
        columns=[f"to_{s}" for s in range(n)])
    trans_df.to_parquet(processed_dir / "hmm_transition_matrix.parquet")

    pd.DataFrame(probs, index=features.index,
                 columns=[f"prob_state_{s}" for s in range(n)]) \
      .to_parquet(processed_dir / "hmm_state_probabilities.parquet")

    features.to_parquet(processed_dir / "hmm_features_raw.parquet")
    pd.DataFrame(X_scaled, index=features.index,
                 columns=features.columns) \
      .to_parquet(processed_dir / "hmm_features_scaled.parquet")

    with open(processed_dir / "hmm_state_labels.json", "w") as f:
        json.dump({str(k): v for k, v in state_labels.items()}, f, indent=2)

    print("\nSaved: hmm_model.pkl | hmm_regime_features.parquet | hmm_state_stats.parquet")
    print("       hmm_transition_matrix.parquet | hmm_state_probabilities.parquet")
    print("       hmm_features_raw.parquet | hmm_features_scaled.parquet | hmm_state_labels.json")
