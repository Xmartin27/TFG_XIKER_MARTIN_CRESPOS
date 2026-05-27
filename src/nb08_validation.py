"""
nb08_validation.py — Walk-forward backtest and validation functions for Notebook 08.

All logic extracted from run_nb08.py so the notebook can call clean functions
and display results with interpretations inline.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import sys
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Last-month prospective validation (NB08 v2) ───────────────────────────────

def regime_context_at(regime_df: pd.DataFrame, target_date, state_labels: dict) -> dict:
    """Return a regime_context-like dict for the row closest to target_date."""
    if regime_df is None or len(regime_df) == 0:
        return {"current_regime_id": None, "current_regime_label": "Unknown",
                "regime_risk_score": 0.35, "prob_transition": 0.025,
                "prob_bear_22d": 0.0,
                "prob_bear_21d": 0.0}
    idx = regime_df.index.asof(pd.Timestamp(target_date))
    row = regime_df.loc[idx]
    state_id = int(row.get("current_regime_id", 0))
    label    = state_labels.get(str(state_id), row.get("current_regime", f"State {state_id}"))
    return {
        "current_regime_id":    state_id,
        "current_regime_label": label,
        "regime_risk_score":    float(row.get("regime_risk_score",
                                              row.get("prob_transition", 0.35))),
        "prob_transition":      float(row.get("prob_transition", 0.025)),
        "prob_bear_22d":        float(row.get("predicted_prob_bear_22d", row.get("predicted_prob_bear_21d", 0.0))),
        "prob_bear_21d":        float(row.get("predicted_prob_bear_22d", row.get("predicted_prob_bear_21d", 0.0))),
    }


def validate_last_month(
    all_recs: dict,
    port_returns: pd.DataFrame,
    horizon: int = 22,
) -> pd.DataFrame:
    """Compute realized returns for each recommended portfolio in the last *horizon* days.

    Parameters
    ----------
    all_recs     : {risk_level: recommendations_df} from recommend_by_risk
    port_returns : daily portfolio returns (columns = portfolio_id as int)
    horizon      : number of trading days in the validation window

    Returns
    -------
    val_df : one row per (risk_level, portfolio) with:
        risk_level, level_name, portfolio_id, realized_return,
        predicted_growth, p_positive, pred_ann_vol, hit, return_positive,
        start_date, end_date
    """
    from src.nb07_recommendation import LEVEL_NAMES_ES

    last_window = port_returns.iloc[-horizon:]
    start_date  = last_window.index[0]
    end_date    = last_window.index[-1]

    rows = []
    for rl, recs in all_recs.items():
        for _, row in recs.iterrows():
            pid = int(row["portfolio_id"])
            if pid in last_window.columns:
                realized = float((1 + last_window[pid].dropna()).prod() - 1)
            else:
                realized = float("nan")

            p_pos    = float(row.get("p_positive", 0.5))
            pred_ret = float(row.get("growth_pred", 0.0))
            hit      = int(realized > 0) if not np.isnan(realized) else None

            rows.append({
                "risk_level":      int(rl),
                "level_name":      LEVEL_NAMES_ES.get(int(rl), f"RL{rl}"),
                "portfolio_id":    pid,
                "realized_return": realized,
                "predicted_growth": pred_ret,
                "p_positive":      p_pos,
                "pred_ann_vol":    float(row.get("pred_ann_vol", float("nan"))),
                "score":           float(row.get("score", float("nan"))),
                "hit":             hit,
                "return_positive": int(realized > 0) if not np.isnan(realized) else None,
                "start_date":      start_date,
                "end_date":        end_date,
            })

    return pd.DataFrame(rows)


def summarise_by_risk(val_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate last-month validation by risk level."""
    agg = val_df.groupby(["risk_level", "level_name"]).agg(
        mean_return   =("realized_return", "mean"),
        std_return    =("realized_return", "std"),
        min_return    =("realized_return", "min"),
        max_return    =("realized_return", "max"),
        hit_rate      =("hit", "mean"),
        n_positive    =("return_positive", "sum"),
        n_portfolios  =("portfolio_id", "count"),
    ).round(4).reset_index()
    return agg


def plot_risk_gradient(
    val_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    regime_label: str,
    fig_dir: Path,
) -> None:
    """3-panel chart validating the risk-return gradient of recommendations.

    Panel 1: Realized return per portfolio, colored by risk level
    Panel 2: Mean realized return vs risk level (bar) with std error bars
    Panel 3: Return std (volatility) vs risk level — should increase with RL
    """
    colors = {1: "#2ca02c", 2: "#98df8a", 3: "#ffbb78", 4: "#ff7f0e", 5: "#d62728"}
    level_names = val_df.groupby("risk_level")["level_name"].first()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: individual portfolio returns by RL
    ax = axes[0]
    for rl, grp in val_df.groupby("risk_level"):
        ax.scatter([rl] * len(grp), grp["realized_return"] * 100,
                   color=colors.get(rl, "grey"), s=120, zorder=3,
                   label=level_names.get(rl, f"RL{rl}"), alpha=0.85)
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_xticks(list(level_names.index))
    ax.set_xticklabels([f"RL{i}\n{level_names[i][:6]}" for i in level_names.index], fontsize=8)
    ax.set_ylabel("Retorno realizado (%)"); ax.set_title("Retorno por nivel de riesgo")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 2: mean return bar chart
    ax = axes[1]
    x  = summary_df["risk_level"].values
    y  = summary_df["mean_return"].values * 100
    ye = summary_df["std_return"].fillna(0).values * 100
    bars = ax.bar(x, y, color=[colors.get(i, "grey") for i in x],
                  alpha=0.85, edgecolor="white", width=0.6)
    ax.errorbar(x, y, yerr=ye, fmt="none", color="black", capsize=5, lw=1.5)
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"RL{i}" for i in x])
    ax.set_ylabel("Retorno medio (%)"); ax.set_title("Retorno medio +/- std por perfil")
    ax.grid(alpha=0.3)
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.1 if val >= 0 else -0.3),
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9)

    # Panel 3: std (volatility) of return per RL — should increase with RL
    ax = axes[2]
    y_std = summary_df["std_return"].fillna(0).values * 100
    ax.bar(x, y_std, color=[colors.get(i, "grey") for i in x],
           alpha=0.85, edgecolor="white", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"RL{i}" for i in x])
    ax.set_ylabel("Volatilidad del retorno (std, %)")
    ax.set_title("Variabilidad del retorno por perfil\n(debe crecer con el riesgo)")
    ax.grid(alpha=0.3)

    plt.suptitle(f"Validacion ultimo mes  [regimen: {regime_label}]", fontsize=13)
    plt.tight_layout()
    out = fig_dir / "08_last_month_validation.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")


# ── Core backtest ─────────────────────────────────────────────────────────────

def forward_return(returns_series: pd.Series, start_idx: int, horizon: int) -> float:
    """Compound return from start_idx over the next horizon days."""
    end_idx = min(start_idx + horizon, len(returns_series))
    window  = returns_series.iloc[start_idx:end_idx]
    return float((1 + window).prod() - 1)


def run_walk_forward_backtest(
    available_pids: List[int],
    port_returns: pd.DataFrame,
    model_results: pd.DataFrame,
    horizon_days: int = 22,
) -> pd.DataFrame:
    """Walk-forward validation: step every horizon_days, evaluate each portfolio.

    For each (portfolio, period) pair the function records:
        - realized_return  : actual compound return over horizon
        - predicted_growth : ENS_growth from model_results
        - predicted_positive: ENS_precision (legacy metadata field)
        - hit              : 1 if realized return over the window is positive
        - return_positive  : 1 if realized_return > 0

    Returns:
        bt_df — one row per (portfolio_id, period)
    """
    total_days   = len(port_returns)
    start_points = list(range(0, total_days - horizon_days, horizon_days))
    rows = []

    for pid in available_pids:
        if pid not in port_returns.columns:
            continue
        ret_series = port_returns[pid].dropna()

        if pid in model_results.index:
            predicted_positive = float(model_results.loc[pid, "ENS_precision"]) \
                if "ENS_precision" in model_results.columns else 0.5
            predicted_growth   = float(model_results.loc[pid, "ENS_growth"]) \
                if "ENS_growth" in model_results.columns else 0.0
        else:
            predicted_positive = 0.5
            predicted_growth   = 0.0

        for start_i in start_points:
            realized = forward_return(ret_series, start_i, horizon_days)
            hit = int(realized > 0)
            rows.append({
                "portfolio_id":       pid,
                "start_date":         port_returns.index[start_i] if start_i < len(port_returns) else None,
                "realized_return":    realized,
                "predicted_growth":   predicted_growth,
                "predicted_positive": predicted_positive,
                "hit":                hit,
                "return_positive":    int(realized > 0),
            })

    bt_df = pd.DataFrame(rows).dropna()
    print(f"Backtest rows: {len(bt_df)}")
    return bt_df


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_validation_metrics(bt_df: pd.DataFrame, horizon_days: int = 22) -> dict:
    """Aggregate backtest metrics: hit rate, annualized return, Sharpe."""
    if len(bt_df) == 0:
        return {}

    overall_hit = float(bt_df["hit"].mean())
    overall_ret = float(bt_df["realized_return"].mean())
    ann_ret     = (1 + overall_ret) ** (252 / horizon_days) - 1
    ret_std     = bt_df["realized_return"].std()
    sharpe      = ann_ret / (ret_std * np.sqrt(252 / horizon_days) + 1e-9)

    metrics = {
        "overall_hit_rate":  overall_hit,
        "mean_realized_ret": overall_ret,
        "annualized_return": ann_ret,
        "backtest_sharpe":   sharpe,
        "n_observations":    len(bt_df),
        "n_portfolios":      bt_df["portfolio_id"].nunique(),
    }
    return metrics


def compute_by_portfolio(bt_df: pd.DataFrame) -> pd.DataFrame:
    """Hit rate, mean return, and count grouped by portfolio_id."""
    return bt_df.groupby("portfolio_id").agg(
        hit_rate=("hit", "mean"),
        mean_return=("realized_return", "mean"),
        n=("hit", "count"),
    ).round(4)


def compute_by_risk(bt_df: pd.DataFrame, recs_df: pd.DataFrame) -> pd.DataFrame:
    """Hit rate and mean return grouped by risk_level (from recommendations)."""
    rec_risk_map   = recs_df.groupby("portfolio_id")["risk_level"].first().to_dict()
    bt              = bt_df.copy()
    bt["risk_level"] = bt["portfolio_id"].map(rec_risk_map)
    return bt.groupby("risk_level", dropna=True).agg(
        hit_rate=("hit", "mean"),
        mean_return=("realized_return", "mean"),
        n=("hit", "count"),
    ).round(4)


def compute_drawdown_stats(bt_df: pd.DataFrame) -> pd.Series:
    """Max drawdown per portfolio over the temporal sequence of bt_df periods."""
    dd = {}
    for pid, grp in bt_df.groupby("portfolio_id"):
        cum   = (1 + grp.sort_values("start_date")["realized_return"]).cumprod()
        dd[pid] = float((cum / cum.cummax() - 1).min())
    return pd.Series(dd)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_validation(
    bt_df: pd.DataFrame,
    metrics: dict,
    by_pid: pd.DataFrame,
    dd_series: pd.Series,
    by_risk: pd.DataFrame,
    source_version: str,
    fig_dir: Path,
) -> None:
    """6-panel validation dashboard."""
    overall_hit = metrics.get("overall_hit_rate", 0)
    overall_ret = metrics.get("mean_realized_ret", 0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Quarterly hit rate
    bt_df = bt_df.copy()
    bt_df["month"] = pd.to_datetime(bt_df["start_date"]).dt.to_period("Q")
    monthly_hr = bt_df.groupby("month")["hit"].mean()
    axes[0, 0].bar(range(len(monthly_hr)), monthly_hr.values, color="#2ca02c", alpha=0.7)
    axes[0, 0].axhline(0.5, color="red", ls="--", lw=1, label="Random (50%)")
    axes[0, 0].axhline(overall_hit, color="blue", ls="-.", lw=1.5, label=f"Mean={overall_hit:.2f}")
    axes[0, 0].set_title("Quarterly Hit Rate"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    # Return distribution
    axes[0, 1].hist(bt_df["realized_return"] * 100, bins=40, color="#1f77b4", edgecolor="white", alpha=0.8)
    axes[0, 1].axvline(0, color="red", ls="--")
    axes[0, 1].axvline(overall_ret * 100, color="green", ls="-.", lw=2)
    axes[0, 1].set_xlabel("Realized Return (%)"); axes[0, 1].set_title("Return Distribution")
    axes[0, 1].grid(alpha=0.3)

    # Predicted vs realized
    axes[0, 2].scatter(bt_df["predicted_growth"] * 100, bt_df["realized_return"] * 100,
                       alpha=0.2, s=5, color="#ff7f0e")
    z  = np.polyfit(bt_df["predicted_growth"].values, bt_df["realized_return"].values, 1)
    xr = np.linspace(bt_df["predicted_growth"].min(), bt_df["predicted_growth"].max(), 100)
    r  = bt_df["predicted_growth"].corr(bt_df["realized_return"])
    axes[0, 2].plot(xr * 100, np.polyval(z, xr) * 100, color="red", lw=2, label=f"r={r:.3f}")
    axes[0, 2].set_xlabel("Predicted Growth (%)"); axes[0, 2].set_ylabel("Realized (%)")
    axes[0, 2].set_title("Predicted vs Realized"); axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # Hit rate by portfolio
    by_pid_sorted = by_pid.sort_values("hit_rate", ascending=True)
    axes[1, 0].barh(range(len(by_pid_sorted)), by_pid_sorted["hit_rate"], color="#2ca02c", alpha=0.7)
    axes[1, 0].axvline(0.5, color="red", ls="--")
    axes[1, 0].set_xlabel("Hit Rate"); axes[1, 0].set_title("Hit Rate by Portfolio"); axes[1, 0].grid(alpha=0.3)

    # Max drawdown
    axes[1, 1].bar(range(len(dd_series)), sorted(dd_series.values), color="#d62728", alpha=0.7)
    axes[1, 1].set_xlabel("Portfolio"); axes[1, 1].set_ylabel("Max Drawdown")
    axes[1, 1].set_title("Max Drawdown by Portfolio"); axes[1, 1].grid(alpha=0.3)

    # Hit rate by risk level
    if len(by_risk) > 0:
        axes[1, 2].bar(by_risk.index.astype(str), by_risk["hit_rate"], color="#9467bd", alpha=0.8)
        axes[1, 2].axhline(0.5, color="red", ls="--")
        axes[1, 2].set_xlabel("Risk Level"); axes[1, 2].set_ylabel("Hit Rate")
        axes[1, 2].set_title("Hit Rate by Risk Profile"); axes[1, 2].grid(alpha=0.3)

    plt.suptitle(f"Recommendation Validation (nb06 {source_version})", fontsize=13)
    plt.tight_layout()
    out = fig_dir / "08_validation_v5.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved: {out}")


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_backtest_outputs(
    bt_df: pd.DataFrame,
    metrics: dict,
    dd_series: pd.Series,
    results_dir: Path,
) -> None:
    """Persist backtest_results_v5.parquet and backtest_summary_v5.json."""
    bt_df.to_parquet(results_dir / "backtest_results_v5.parquet", index=False)

    summary = {
        "source_version":    metrics.get("source_version", "unknown"),
        "overall_hit_rate":  float(metrics.get("overall_hit_rate", 0)),
        "mean_realized_ret": float(metrics.get("mean_realized_ret", 0)),
        "annualized_return": float(metrics.get("annualized_return", 0)),
        "backtest_sharpe":   float(metrics.get("backtest_sharpe", 0)),
        "n_observations":    int(metrics.get("n_observations", 0)),
        "n_portfolios":      int(metrics.get("n_portfolios", 0)),
        "mean_max_drawdown": float(dd_series.mean()) if len(dd_series) > 0 else 0.0,
    }
    with open(results_dir / "backtest_summary_v5.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[OK] backtest_results_v5.parquet")
    print("[OK] backtest_summary_v5.json")


# ── Crisis backtesting ────────────────────────────────────────────────────────

CRISIS_WINDOWS: Dict[str, tuple] = {
    "COVID-2020": ("2020-01-15", "2020-06-30"),
    "GFC-2008":   ("2008-09-01", "2009-03-31"),
}


def run_crisis_backtest(
    bt_df: pd.DataFrame,
    crisis_name: str,
    crisis_start: str,
    crisis_end: str,
    horizon_days: int = 22,
) -> dict:
    """Compute validation metrics for a specific crisis window.

    Subsets bt_df (already computed full backtest) to the crisis period
    and runs compute_validation_metrics on that slice.
    Also computes max drawdown within the crisis window.
    """
    t0 = pd.Timestamp(crisis_start)
    t1 = pd.Timestamp(crisis_end)
    mask = (pd.to_datetime(bt_df["start_date"]) >= t0) & \
           (pd.to_datetime(bt_df["start_date"]) <= t1)
    sub = bt_df[mask].copy()

    if len(sub) == 0:
        print(f"  [{crisis_name}] No data in window {crisis_start} - {crisis_end}")
        return {"crisis": crisis_name, "n_observations": 0}

    m = compute_validation_metrics(sub, horizon_days)
    dd = compute_drawdown_stats(sub)
    m["crisis"]          = crisis_name
    m["crisis_start"]    = crisis_start
    m["crisis_end"]      = crisis_end
    m["mean_max_drawdown"] = float(dd.mean()) if len(dd) > 0 else 0.0
    print(f"  [{crisis_name}] n={m['n_observations']}  hit={m['overall_hit_rate']:.3f}"
          f"  ann_ret={m['annualized_return']*100:.1f}%  maxDD={m['mean_max_drawdown']*100:.1f}%")
    return m


def run_crisis_backtest_by_risk(
    bt_df: pd.DataFrame,
    recs_df: pd.DataFrame,
    crisis_name: str,
    crisis_start: str,
    crisis_end: str,
    horizon_days: int = 22,
) -> pd.DataFrame:
    """Compute crisis metrics broken down by risk level."""
    t0 = pd.Timestamp(crisis_start)
    t1 = pd.Timestamp(crisis_end)
    mask = (pd.to_datetime(bt_df["start_date"]) >= t0) & \
           (pd.to_datetime(bt_df["start_date"]) <= t1)
    sub = bt_df[mask].copy()

    if len(sub) == 0:
        return pd.DataFrame()

    rec_risk_map = recs_df.groupby("portfolio_id")["risk_level"].first().to_dict()
    sub["risk_level"] = sub["portfolio_id"].map(rec_risk_map)

    rows = []
    for rl, grp in sub.groupby("risk_level", dropna=True):
        m   = compute_validation_metrics(grp, horizon_days)
        dd  = compute_drawdown_stats(grp)
        rows.append({
            "risk_level":      int(rl),
            "crisis":          crisis_name,
            "hit_rate":        m.get("overall_hit_rate", np.nan),
            "ann_return":      m.get("annualized_return", np.nan),
            "mean_max_dd":     float(dd.mean()) if len(dd) > 0 else np.nan,
            "n_obs":           m.get("n_observations", 0),
        })
    return pd.DataFrame(rows).sort_values("risk_level")


def plot_crisis_comparison(
    full_by_risk: pd.DataFrame,
    crisis_results: Dict[str, pd.DataFrame],
    fig_dir: Path,
) -> None:
    """Bar chart: hit rate and annualized return by risk level, full vs crisis periods."""
    colors_period = {
        "Full period": "#1f77b4",
        "COVID-2020":  "#d62728",
        "GFC-2008":    "#ff7f0e",
    }
    risk_levels = sorted(full_by_risk["risk_level"].unique()) if len(full_by_risk) > 0 else [1,2,3,4,5]
    n_rl  = len(risk_levels)
    periods = ["Full period"] + list(crisis_results.keys())
    n_periods = len(periods)
    bar_w = 0.8 / n_periods
    x = np.arange(n_rl)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, (metric, ylabel, title) in enumerate([
        ("hit_rate",   "Hit Rate",              "Hit Rate por perfil y periodo"),
        ("ann_return", "Retorno anualizado (%)", "Retorno anualizado por perfil y periodo"),
    ]):
        ax = axes[ax_idx]
        for p_idx, period in enumerate(periods):
            if period == "Full period":
                df_p = full_by_risk.copy()
                df_p["ann_return"] = df_p.get("ann_return",
                    df_p.get("mean_return", pd.Series(dtype=float)))
            else:
                df_p = crisis_results.get(period, pd.DataFrame())
            if len(df_p) == 0:
                continue

            vals = []
            for rl in risk_levels:
                row = df_p[df_p["risk_level"] == rl]
                v = float(row[metric].values[0]) if len(row) > 0 else np.nan
                vals.append(v * 100 if metric == "ann_return" else v)

            offset = (p_idx - n_periods / 2 + 0.5) * bar_w
            ax.bar(x + offset, vals, width=bar_w,
                   label=period, color=colors_period.get(period, "grey"),
                   alpha=0.85, edgecolor="white")

        if metric == "hit_rate":
            ax.axhline(0.5, color="black", ls="--", lw=1, label="50% (random)")
        else:
            ax.axhline(0.0, color="black", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"RL{rl}" for rl in risk_levels])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Comportamiento en periodos de crisis vs periodo completo", fontsize=12)
    plt.tight_layout()
    out = fig_dir / "08_crisis_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def save_backtest_outputs_v5(
    bt_df: pd.DataFrame,
    metrics: dict,
    dd_series: pd.Series,
    crisis_metrics: dict,
    results_dir: Path,
) -> None:
    """Persist backtest_results_v5.parquet and backtest_summary_v5.json with crisis section."""
    bt_df.to_parquet(results_dir / "backtest_results_v5.parquet", index=False)

    summary = {
        "source_version":    metrics.get("source_version", "unknown"),
        "overall_hit_rate":  float(metrics.get("overall_hit_rate", 0)),
        "mean_realized_ret": float(metrics.get("mean_realized_ret", 0)),
        "annualized_return": float(metrics.get("annualized_return", 0)),
        "backtest_sharpe":   float(metrics.get("backtest_sharpe", 0)),
        "n_observations":    int(metrics.get("n_observations", 0)),
        "n_portfolios":      int(metrics.get("n_portfolios", 0)),
        "mean_max_drawdown": float(dd_series.mean()) if len(dd_series) > 0 else 0.0,
        "crisis_periods":    {
            name: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in cm.items()}
            for name, cm in crisis_metrics.items()
        },
    }
    with open(results_dir / "backtest_summary_v5.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[OK] backtest_results_v5.parquet")
    print("[OK] backtest_summary_v5.json (con crisis_periods)")


def compare_recommendation_versions(
    bt_new: pd.DataFrame,
    recs_baseline: pd.DataFrame,
    recs_new: pd.DataFrame,
) -> pd.DataFrame:
    """Compare backtest hit-rate and annualized return per risk level.

    bt_new contains the walk-forward results for the NEW recommendation version.
    recs_baseline provides portfolio IDs for the ENS baseline (used to label the comparison).
    Returns a DataFrame with one row per risk level and columns for both versions.
    """
    rows = []
    for rl in sorted(bt_new["risk_level"].unique()):
        sub = bt_new[bt_new["risk_level"] == rl]
        hit_new = float(sub["hit"].mean()) if len(sub) > 0 else float("nan")
        ann_new = float(sub["realized_return"].mean() * (252 / 22)) if len(sub) > 0 else float("nan")
        pids_new = sorted(recs_new[recs_new["risk_level"] == rl]["portfolio_id"].unique().tolist())

        pids_base = sorted(recs_baseline[recs_baseline["risk_level"] == rl]["portfolio_id"].unique().tolist()) \
                    if recs_baseline is not None else []

        rows.append({
            "risk_level":         int(rl),
            "hit_rate_new":       round(hit_new, 4),
            "ann_return_new":     round(ann_new, 4),
            "portfolios_new":     str(pids_new),
            "portfolios_baseline": str(pids_base),
            "portfolios_changed": pids_new != pids_base,
        })

    return pd.DataFrame(rows).set_index("risk_level")
