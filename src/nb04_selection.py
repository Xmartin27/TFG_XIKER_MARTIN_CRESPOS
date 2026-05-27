"""
nb04_selection.py — Portfolio selection via K-Means clustering and composite scoring.
"""
import json
import warnings
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import sys
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCORE_COLS   = ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "neg_max_drawdown"]
SCORE_WEIGHTS = [0.40, 0.30, 0.20, 0.10]
MIN_K = 40
MAX_K = 50


def standardize_features(features: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """Standardize numeric feature columns. Returns (X_scaled, scaler, feature_cols)."""
    feature_cols = [c for c in features.columns if c != "portfolio_id"]
    X_raw = np.nan_to_num(features[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    return X, scaler, feature_cols


def run_pca(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """2-component PCA projection for visualization."""
    pca   = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.round(3)}")
    return X_pca, pca


def select_optimal_k(X: np.ndarray, k_values: List[int] = None) -> Tuple[int, pd.DataFrame]:
    """Elbow + silhouette analysis to pick optimal K within [MIN_K, MAX_K].

    Returns (optimal_k, metrics_df).
    """
    if k_values is None:
        k_values = list(range(5, 51, 5))

    records = []
    for k in k_values:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        lbl = km.fit_predict(X)
        sil = silhouette_score(X, lbl)
        db  = davies_bouldin_score(X, lbl)
        records.append({"k": k, "inertia": km.inertia_, "silhouette": sil, "db_score": db})
        print(f"  K={k:3d}: Sil={sil:.4f}  DB={db:.4f}")

    metrics_df = pd.DataFrame(records)
    inertias   = metrics_df["inertia"].values
    x = np.array(k_values, dtype=float)
    xn = (x - x.min()) / max(x.max() - x.min(), 1e-9)
    yn = (inertias - inertias.min()) / max(inertias.max() - inertias.min(), 1e-9)
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    line   = p2 - p1; lnorm = np.linalg.norm(line) + 1e-12
    distances = [float(np.abs(np.cross(line, np.array([xn[i], yn[i]]) - p1)) / lnorm)
                 for i in range(len(xn))]
    elbow_k   = int(k_values[int(np.argmax(distances))])
    optimal_k = max(min(elbow_k, MAX_K), MIN_K)
    print(f"Elbow K={elbow_k}  Selected K={optimal_k}")
    return optimal_k, metrics_df


def fit_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    """Fit final K-Means and return cluster labels."""
    km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    labels = km.fit_predict(X)
    print(f"K-Means K={k}  Inertia={km.inertia_:.1f}")
    return labels


def compute_composite_scores(features: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """Compute z-score weighted composite score per portfolio."""
    score_df = features[["portfolio_id", "sharpe_ratio", "sortino_ratio",
                          "calmar_ratio", "max_drawdown"]].copy()
    score_df["neg_max_drawdown"] = -score_df["max_drawdown"]
    for col in SCORE_COLS:
        mu, std = score_df[col].mean(), score_df[col].std()
        score_df[f"{col}_z"] = (score_df[col] - mu) / (std + 1e-9)
    score_df["composite_score"] = sum(
        w * score_df[f"{c}_z"] for c, w in zip(SCORE_COLS, SCORE_WEIGHTS)
    )
    score_df["cluster"] = cluster_labels
    print(f"Composite score: [{score_df.composite_score.min():.4f}, {score_df.composite_score.max():.4f}]")
    return score_df


def select_portfolios(score_df: pd.DataFrame, k: int) -> Tuple[List[int], pd.DataFrame]:
    """Select best portfolio per cluster + global optimal.

    Returns (selected_ids, details_df).
    """
    selected_ids, details = [], []
    for cl in range(k):
        cl_df = score_df[score_df["cluster"] == cl]
        best  = cl_df.loc[cl_df["composite_score"].idxmax()]
        pid   = int(best["portfolio_id"])
        selected_ids.append(pid)
        details.append({
            "cluster": cl, "portfolio_id": pid,
            "sharpe_ratio":      float(best["sharpe_ratio"]),
            "sortino_ratio":     float(best["sortino_ratio"]),
            "calmar_ratio":      float(best["calmar_ratio"]),
            "neg_max_drawdown":  float(best["neg_max_drawdown"]),
            "composite_score":   float(best["composite_score"]),
        })
        print(f"  Cluster {cl:2d}: pid={pid}  score={best['composite_score']:.4f}")

    global_best = score_df.loc[score_df["composite_score"].idxmax()]
    global_pid  = int(global_best["portfolio_id"])
    if global_pid not in selected_ids:
        selected_ids.append(global_pid)
        details.append({"cluster": -1, "portfolio_id": global_pid,
                         "composite_score": float(global_best["composite_score"])})
        print(f"  Global optimal: pid={global_pid} added")
    else:
        print(f"  Global optimal pid={global_pid} already selected")

    print(f"Total selected: {len(selected_ids)} portfolios")
    return selected_ids, pd.DataFrame(details)


def plot_selection(
    features: pd.DataFrame,
    X_pca: np.ndarray,
    cluster_labels: np.ndarray,
    selected_ids: List[int],
    fig_dir: Path,
) -> None:
    """PCA cluster plot + risk-return scatter with selected portfolios highlighted."""
    sel_mask  = features["portfolio_id"].isin(selected_ids)
    sel_idx   = [features[features["portfolio_id"] == pid].index[0] for pid in selected_ids
                 if pid in features["portfolio_id"].values]
    sel_feats = features[sel_mask]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="tab20", s=5, alpha=0.3)
    if sel_idx:
        axes[0].scatter(X_pca[sel_idx, 0], X_pca[sel_idx, 1],
                        c="red", s=120, marker="*", edgecolor="black", lw=0.5,
                        zorder=5, label="Óptimas")
    axes[0].set_title(f"PCA: Clusters y carteras óptimas"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].scatter(features.annual_volatility * 100, features.annual_return * 100,
                    c="lightgrey", s=5, alpha=0.3, label="Todas")
    axes[1].scatter(sel_feats.annual_volatility * 100, sel_feats.annual_return * 100,
                    c="red", s=120, marker="*", edgecolor="black", lw=0.5,
                    zorder=5, label=f"Seleccionadas (n={len(selected_ids)})")
    axes[1].set_xlabel("Volatilidad (%)"); axes[1].set_ylabel("Rentabilidad (%)")
    axes[1].set_title("Riesgo-Rentabilidad: Carteras seleccionadas"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = fig_dir / "04_cluster_selection_v4.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    return fig


def save_outputs(
    selected_ids: List[int],
    details_df: pd.DataFrame,
    composition: pd.DataFrame,
    cluster_labels: np.ndarray,
    features: pd.DataFrame,
    scaler: StandardScaler,
    processed_dir: Path,
) -> None:
    """Persist all NB04 outputs."""
    sel_comp = composition[composition["portfolio_id"].isin(selected_ids)]
    sel_comp.to_parquet(processed_dir / "selected_portfolios.parquet", index=False)
    sel_comp.to_parquet(processed_dir / "selected_portfolios_optimal_v4.parquet", index=False)
    with open(processed_dir / "selected_portfolios_ids.json", "w") as f:
        json.dump(selected_ids, f, indent=2)
    with open(processed_dir / "selected_portfolios_ids_optimal_v4.json", "w") as f:
        json.dump(selected_ids, f, indent=2)
    cluster_df = pd.DataFrame({"portfolio_id": features["portfolio_id"].values,
                                "cluster": cluster_labels})
    cluster_df.to_parquet(processed_dir / "cluster_labels.parquet", index=False)
    details_df.to_parquet(processed_dir / "optimal_selection_details_v4.parquet", index=False)
    joblib.dump(scaler, processed_dir / "scaler_portfolio_features.pkl")
    print("Saved: selected_portfolios.parquet | selected_portfolios_ids.json")
    print("       cluster_labels.parquet | optimal_selection_details_v4.parquet")


def _detail_row(row: "pd.Series", max_corr_at_sel: float) -> dict:
    return {
        "cluster":               int(row.get("cluster", -1)),
        "portfolio_id":          int(row["portfolio_id"]),
        "sharpe_ratio":          float(row.get("sharpe_ratio", 0)),
        "sortino_ratio":         float(row.get("sortino_ratio", 0)),
        "calmar_ratio":          float(row.get("calmar_ratio", 0)),
        "neg_max_drawdown":      float(row.get("neg_max_drawdown", 0)),
        "composite_score":       float(row["composite_score"]),
        "max_corr_at_selection": max_corr_at_sel,
    }


def select_portfolios_corr_constrained(
    score_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    max_corr: float = 0.65,
    target_n: int = 40,
) -> Tuple[List[int], pd.DataFrame]:
    """Greedy score-ranked, correlation-constrained portfolio selection.

    Sort all candidates by composite_score descending. Add each candidate only
    if its max absolute Pearson correlation with already-selected portfolios is
    below max_corr. Stop when target_n portfolios are selected.

    returns_df: wide DataFrame (index=date, columns=portfolio_id as int).
    """
    sorted_df = score_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    avail_ret  = set(returns_df.columns)

    selected_ids: List[int] = []
    details = []

    for _, row in sorted_df.iterrows():
        if len(selected_ids) >= target_n:
            break
        pid = int(row["portfolio_id"])

        if not selected_ids:
            selected_ids.append(pid)
            details.append(_detail_row(row, float("nan")))
            print(f"  + pid={pid}  score={row['composite_score']:.4f}  (first)")
            continue

        sel_avail = [s for s in selected_ids if s in avail_ret]
        if pid not in avail_ret or not sel_avail:
            selected_ids.append(pid)
            details.append(_detail_row(row, float("nan")))
            continue

        cand_ret = returns_df[pid].dropna()
        max_c = float(
            returns_df[sel_avail].reindex(cand_ret.index).corrwith(cand_ret).abs().max()
        )

        if max_c < max_corr:
            selected_ids.append(pid)
            details.append(_detail_row(row, max_c))
            print(f"  + pid={pid}  score={row['composite_score']:.4f}  max_corr={max_c:.3f}")
        else:
            print(f"  - pid={pid} rejected (max_corr={max_c:.3f} >= {max_corr})")

    n = len(selected_ids)
    print(f"\nGreedy result: {n}/{target_n} portfolios  max_corr<={max_corr}")
    if n < target_n:
        print(f"  WARNING: only {n} portfolios found — consider relaxing max_corr")
    return selected_ids, pd.DataFrame(details)


def _greedy_from_pool(
    pool_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    avail_ret: set,
    max_corr: float,
    n_per_tier: int,
    tier_label: str,
) -> Tuple[List[int], List[dict]]:
    """Greedy corr-constrained selection within a single volatility tier pool."""
    pool_sorted = pool_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    sel: List[int] = []
    dets = []
    for _, row in pool_sorted.iterrows():
        if len(sel) >= n_per_tier:
            break
        pid = int(row["portfolio_id"])
        if not sel or pid not in avail_ret:
            sel.append(pid)
            d = _detail_row(row, float("nan"))
            d["vol_tier"] = tier_label
            dets.append(d)
            continue
        sel_avail = [s for s in sel if s in avail_ret]
        if not sel_avail:
            sel.append(pid)
            d = _detail_row(row, float("nan"))
            d["vol_tier"] = tier_label
            dets.append(d)
            continue
        cand_ret = returns_df[pid].dropna()
        max_c = float(
            returns_df[sel_avail].reindex(cand_ret.index).corrwith(cand_ret).abs().max()
        )
        if max_c < max_corr:
            sel.append(pid)
            d = _detail_row(row, max_c)
            d["vol_tier"] = tier_label
            dets.append(d)
    return sel, dets


def select_portfolios_tiered(
    score_df: pd.DataFrame,
    features: pd.DataFrame,
    returns_df: pd.DataFrame,
    n_tiers: int = 5,
    n_per_tier: int = 8,
    max_corr: float = 0.85,
) -> Tuple[List[int], pd.DataFrame]:
    """Volatility-tiered, correlation-constrained portfolio selection.

    Divides all portfolios into n_tiers volatility quantile bands, then within
    each band selects the n_per_tier highest-scoring portfolios whose pairwise
    Pearson |corr| with already-selected portfolios in that tier is < max_corr.

    Total selected: n_tiers * n_per_tier (default 5 * 8 = 40).
    Guarantees representation at every volatility level so N07 has distinct
    candidates per risk pool.
    """
    merged = score_df.merge(
        features[["portfolio_id", "annual_volatility"]],
        on="portfolio_id", how="left",
    )
    merged["vol_tier"] = pd.qcut(
        merged["annual_volatility"], q=n_tiers,
        labels=[f"T{i+1}" for i in range(n_tiers)],
    )

    avail_ret = set(returns_df.columns)
    all_ids: List[int] = []
    all_dets = []

    for tier in [f"T{i+1}" for i in range(n_tiers)]:
        pool = merged[merged["vol_tier"] == tier]
        sel, dets = _greedy_from_pool(
            pool, returns_df, avail_ret, max_corr, n_per_tier, tier,
        )
        print(f"  {tier}: {len(sel)}/{n_per_tier} selected  (pool size {len(pool)})")
        if len(sel) < n_per_tier:
            print(f"    WARNING: only {len(sel)} found in {tier} — relaxing max_corr to 0.95")
            extra, extra_dets = _greedy_from_pool(
                pool, returns_df, avail_ret, 0.95, n_per_tier - len(sel), tier,
            )
            for pid in extra:
                if pid not in sel:
                    sel.append(pid)
            dets.extend(d for d in extra_dets if d["portfolio_id"] not in [x["portfolio_id"] for x in dets])
        all_ids.extend(sel)
        all_dets.extend(dets)

    print(f"\nTiered result: {len(all_ids)} portfolios  ({n_tiers} tiers x {n_per_tier}/tier)")
    return all_ids, pd.DataFrame(all_dets)
