"""
nb06_prediction.py — Load and analyse nb06 model results.

NB06 training (ElasticNet, LightGBM, CatBoost, GARCH volatility) takes 60+ minutes
and is run via run_nb06.py. This module provides functions to LOAD and DISPLAY the
pre-computed results so the notebook can show them without retraining.
"""
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sys
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODEL_COLS = {
    "M1": ["M1_growth", "M1_mae", "M1_rmse", "M1_mape_pct", "M1_dir_hit", "M1_precision"],
    "M2": ["M2_growth", "M2_mae", "M2_rmse", "M2_mape_pct", "M2_dir_hit", "M2_precision"],
    "M3": ["M3_growth", "M3_mae", "M3_rmse", "M3_mape_pct", "M3_dir_hit", "M3_precision"],
    "M4": ["M4_growth", "M4_mae", "M4_rmse", "M4_mape_pct", "M4_dir_hit", "M4_precision"],
    "ENS": ["ENS_growth", "ENS_mae", "ENS_rmse", "ENS_mape_pct",
            "ENS_dir_hit", "ENS_precision", "ENS_sharpe", "ENS_maxdd", "ENS_sortino"],
}
VOL_MODELS = ["HARQ", "EWMA", "GARCH", "EGARCH", "XGB_vol", "RF_vol", "LGB_vol"]


def load_model_results(results_dir: Path) -> Tuple[pd.DataFrame, str]:
    """Load model_results parquet (priority: final > v15 > v14 > v13)."""
    for tag, fname in [("final", "nb06_h22_model_results.parquet"),
                       ("v15",   "nb06v15_h22_model_results.parquet"),
                       ("v14",   "nb06v14_h22_model_results.parquet"),
                       ("v13",   "nb06v13_h22_model_results.parquet")]:
        p = results_dir / fname
        if p.exists():
            df = pd.read_parquet(p)
            print(f"Loaded model results ({tag}): {df.shape}")
            return df, tag
    raise FileNotFoundError("No nb06 model results found. Run run_nb06.py first.")


def load_volatility_predictions(results_dir: Path) -> Optional[pd.DataFrame]:
    """Load volatility predictions if available."""
    for fname in ["nb06_volatility_predictions.parquet",
                  "nb06v15_volatility_predictions.parquet",
                  "nb06v14_volatility_predictions.parquet"]:
        p = results_dir / fname
        if p.exists():
            df = pd.read_parquet(p)
            print(f"Loaded volatility predictions: {df.shape}  models={df['model'].unique().tolist()}")
            return df
    print("Volatility predictions not found.")
    return None


def load_summary(results_dir: Path) -> Optional[pd.DataFrame]:
    """Load per-portfolio summary if available."""
    for fname in ["nb06v15_h22_summary.parquet", "nb06v14_h22_summary.parquet",
                  "nb06_h22_summary.parquet"]:
        p = results_dir / fname
        if p.exists():
            df = pd.read_parquet(p)
            print(f"Loaded summary: {df.shape}")
            return df
    return None


def model_metrics_table(model_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean metrics across portfolios for each model."""
    rows = []
    for model, cols in MODEL_COLS.items():
        avail = [c for c in cols if c in model_results.columns]
        if not avail:
            continue
        row = {"model": model}
        for c in avail:
            row[c.replace(f"{model}_", "")] = float(model_results[c].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def plot_model_comparison(model_results: pd.DataFrame, fig_dir: Path) -> None:
    """Bar plots: directional hit rate and MAPE per model across portfolios."""
    metrics = model_metrics_table(model_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Directional hit rate
    hit_cols = {m: f"{m}_dir_hit" for m in MODEL_COLS if f"{m}_dir_hit" in model_results.columns}
    if hit_cols:
        hit_means = {m: model_results[c].mean() for m, c in hit_cols.items()}
        axes[0].bar(hit_means.keys(), [v * 100 for v in hit_means.values()],
                    color="#2ca02c", alpha=0.8, edgecolor="white")
        axes[0].axhline(50, color="red", ls="--", lw=1, label="Azar (50%)")
        axes[0].set_ylabel("Tasa de acierto direccional (%)"); axes[0].set_ylim(0, 80)
        axes[0].set_title("Acierto direccional por modelo"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # MAPE
    mape_cols = {m: f"{m}_mape_pct" for m in MODEL_COLS if f"{m}_mape_pct" in model_results.columns}
    if mape_cols:
        mape_means = {m: model_results[c].mean() for m, c in mape_cols.items()}
        axes[1].bar(mape_means.keys(), list(mape_means.values()),
                    color="#1f77b4", alpha=0.8, edgecolor="white")
        axes[1].set_ylabel("MAPE (%)"); axes[1].set_title("MAPE por modelo"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = fig_dir / "06_model_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    return fig


def plot_ensemble_distribution(model_results: pd.DataFrame, fig_dir: Path) -> None:
    """Distribution of ENS precision and predicted growth."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "ENS_precision" in model_results.columns:
        axes[0].hist(model_results["ENS_precision"] * 100, bins=20,
                     color="#ff7f0e", edgecolor="white", alpha=0.8)
        axes[0].axvline(50, color="red", ls="--", label="50%")
        axes[0].set_xlabel("Precisión direccional ENS (%)"); axes[0].legend()
        axes[0].set_title("Distribución precisión del ensemble"); axes[0].grid(alpha=0.3)

    if "ENS_growth" in model_results.columns:
        axes[1].hist(model_results["ENS_growth"] * 100, bins=20,
                     color="#9467bd", edgecolor="white", alpha=0.8)
        axes[1].axvline(0, color="red", ls="--")
        axes[1].set_xlabel("Crecimiento predicho ENS (%)");
        axes[1].set_title("Distribución crecimiento predicho"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = fig_dir / "06_ensemble_dist.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    return fig


def plot_volatility_models(vol_df: pd.DataFrame, fig_dir: Path) -> None:
    """Mean predicted vol per model across portfolios."""
    if vol_df is None:
        print("No volatility data to plot.")
        return
    last_fold = vol_df["fold"].max()
    sub = vol_df[vol_df["fold"] == last_fold]
    mean_vol = (sub.groupby("model")["sigma2_pred"].mean()
                   .apply(lambda x: np.sqrt(252 * max(x, 0))) * 100)
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_vol.sort_values().plot.bar(ax=ax, color="#d62728", alpha=0.8, edgecolor="white")
    ax.set_ylabel("Volatilidad anualizada predicha (%)"); ax.set_title("Modelos de volatilidad: predicción media")
    ax.grid(alpha=0.3); plt.tight_layout()
    out = fig_dir / "06_volatility_models.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    return fig


def top_portfolios_table(model_results: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N portfolios by ENS precision."""
    cols = ["ENS_precision", "ENS_growth", "ENS_sharpe", "ENS_maxdd"]
    avail = [c for c in cols if c in model_results.columns]
    if not avail:
        return pd.DataFrame()
    return model_results[avail].nlargest(n, avail[0]).round(4)
