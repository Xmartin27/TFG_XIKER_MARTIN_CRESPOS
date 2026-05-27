"""
nb02_portfolios.py — Random portfolio generation with stability constraints.
"""
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sys
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

N_PORTFOLIOS = 4000
N_CANDIDATES = 12000
MIN_ASSETS   = 5
MAX_ASSETS   = 15
MIN_WEIGHT   = 0.02
MAX_WEIGHT   = 0.35
MIN_SHARPE   = 0.30
MAX_VOL      = 0.20
TRADING_DAYS = 252


def make_portfolio(tickers: List[str], rng: np.random.Generator) -> Dict[str, float]:
    """Generate one random portfolio: 5-15 assets, Dirichlet weights clipped to [0.02, 0.35]."""
    n      = rng.integers(MIN_ASSETS, MAX_ASSETS + 1)
    chosen = rng.choice(tickers, size=n, replace=False)
    w      = rng.dirichlet(np.ones(n))
    w      = np.clip(w, MIN_WEIGHT, MAX_WEIGHT)
    w      = w / w.sum()
    return dict(zip(chosen, w))


def portfolio_features(
    weights_dict: Dict[str, float],
    ret_df: pd.DataFrame,
    mkt_ret: pd.Series,
    universe_meta: dict,
    td: int = TRADING_DAYS,
) -> dict:
    """Compute comprehensive portfolio metrics.

    Returns:
        dict with annual_return, annual_volatility, sharpe_ratio, max_drawdown,
        sortino_ratio, calmar_ratio, skewness, kurtosis, beta, n_assets, hhi,
        pct_equity, pct_fixed_income, pct_commodities, pct_reits.
    """
    w  = pd.Series(weights_dict).reindex(ret_df.columns).fillna(0.0)
    w  = w / w.sum()
    pr = (ret_df * w).sum(axis=1)

    ann_ret = pr.mean() * td
    ann_vol = pr.std() * np.sqrt(td)
    sharpe  = ann_ret / (ann_vol + 1e-9)

    cum   = (1 + pr).cumprod()
    maxdd = (cum / cum.cummax() - 1).min()

    neg     = pr[pr < 0]
    sortino = ann_ret / (neg.std() * np.sqrt(td) + 1e-9)
    calmar  = ann_ret / (abs(maxdd) + 1e-9)

    cov_mat = np.cov(pr.values, mkt_ret.reindex(pr.index).fillna(0).values)
    beta    = cov_mat[0, 1] / (cov_mat[1, 1] + 1e-9)

    w_vals = np.array(list(weights_dict.values()))
    hhi    = (w_vals ** 2).sum()

    broad   = universe_meta.get("broad_categories", {})
    classes = ["Equity", "Fixed Income", "Commodities", "REITs"]
    exp = {
        f"pct_{c.lower().replace(' ', '_')}":
        sum(weights_dict.get(t, 0) for t in weights_dict if broad.get(t) == c)
        for c in classes
    }
    return {
        "annual_return":     ann_ret,
        "annual_volatility": ann_vol,
        "sharpe_ratio":      sharpe,
        "max_drawdown":      maxdd,
        "sortino_ratio":     sortino,
        "calmar_ratio":      calmar,
        "skewness":          float(pr.skew()),
        "kurtosis":          float(pr.kurtosis()),
        "beta":              beta,
        "n_assets":          len(weights_dict),
        "hhi":               hhi,
        **exp,
    }


def generate_stable_portfolios(
    tickers: List[str],
    returns: pd.DataFrame,
    market_ret: pd.Series,
    universe_meta: dict,
    n_candidates: int = N_CANDIDATES,
    n_portfolios: int = N_PORTFOLIOS,
    min_sharpe: float = MIN_SHARPE,
    max_vol: float = MAX_VOL,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[Dict[str, float]]]:
    """Generate n_candidates random portfolios, apply stability filter, return top n_portfolios by Sharpe.

    Returns:
        feat_df            — DataFrame of portfolio metrics
        valid_portfolios   — list of weight dicts corresponding to feat_df rows
    """
    rng = np.random.default_rng(seed)
    all_portfolios = [make_portfolio(tickers, rng) for _ in range(n_candidates)]
    print(f"Generated {len(all_portfolios)} candidate portfolios")

    features_list, valid_portfolios = [], []
    for i, pw in enumerate(all_portfolios):
        if i % 2000 == 0:
            print(f"  Processing {i}/{n_candidates}...")
        feats = portfolio_features(pw, returns, market_ret, universe_meta)
        if feats["sharpe_ratio"] >= min_sharpe and feats["annual_volatility"] <= max_vol:
            features_list.append(feats)
            valid_portfolios.append(pw)

    print(f"Passed stability filter: {len(valid_portfolios)} / {n_candidates}")

    feat_df = pd.DataFrame(features_list)
    feat_df["portfolio_id"] = range(1, len(feat_df) + 1)
    feat_df = feat_df.sort_values("sharpe_ratio", ascending=False).head(n_portfolios).reset_index(drop=True)
    feat_df["portfolio_id"] = range(1, len(feat_df) + 1)

    selected_idx   = feat_df.index.tolist()
    valid_portfolios = [valid_portfolios[i] for i in selected_idx]

    print(f"Selected {len(feat_df)} stable portfolios")
    print(f"  Sharpe: [{feat_df.sharpe_ratio.min():.3f}, {feat_df.sharpe_ratio.max():.3f}]")
    print(f"  Vol:    [{feat_df.annual_volatility.min():.3f}, {feat_df.annual_volatility.max():.3f}]")
    return feat_df, valid_portfolios


def build_composition(feat_df: pd.DataFrame, portfolios: List[Dict[str, float]]) -> pd.DataFrame:
    """Long-format composition table: portfolio_id, ticker, weight."""
    rows = []
    for pid, pw in zip(feat_df["portfolio_id"], portfolios):
        for ticker, weight in pw.items():
            rows.append({"portfolio_id": pid, "ticker": ticker, "weight": weight})
    return pd.DataFrame(rows)


def build_returns_matrix(
    feat_df: pd.DataFrame,
    portfolios: List[Dict[str, float]],
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Daily portfolio returns matrix: (T × n_portfolios)."""
    series = []
    for pid, pw in zip(feat_df["portfolio_id"], portfolios):
        w  = pd.Series(pw).reindex(returns.columns).fillna(0.0)
        w  = w / w.sum()
        pr = (returns * w).sum(axis=1)
        pr.name = pid
        series.append(pr)
    mat = pd.concat(series, axis=1)
    print(f"Portfolio returns matrix: {mat.shape}")
    return mat


def plot_portfolios(feat_df: pd.DataFrame, fig_dir: Path) -> None:
    """Risk-return scatter + Sharpe distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc = axes[0].scatter(feat_df.annual_volatility * 100, feat_df.annual_return * 100,
                         c=feat_df.sharpe_ratio, cmap="RdYlGn", s=5, alpha=0.5)
    plt.colorbar(sc, ax=axes[0], label="Sharpe")
    axes[0].set_xlabel("Volatilidad anual (%)"); axes[0].set_ylabel("Rentabilidad anual (%)")
    axes[0].set_title(f"Frontera eficiente: {len(feat_df)} carteras estables")
    axes[0].grid(alpha=0.3)

    axes[1].hist(feat_df.sharpe_ratio, bins=40, color="#2ca02c", edgecolor="white", alpha=0.8)
    axes[1].axvline(feat_df.sharpe_ratio.mean(), color="red", ls="--",
                    label=f"Media={feat_df.sharpe_ratio.mean():.2f}")
    axes[1].set_xlabel("Sharpe Ratio"); axes[1].set_title("Distribución del Sharpe")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = fig_dir / "02_portfolios_v3.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    return fig


def save_outputs(
    feat_df: pd.DataFrame,
    composition: pd.DataFrame,
    port_returns: pd.DataFrame,
    processed_dir: Path,
) -> None:
    """Persist all NB02 outputs."""
    feat_cols = ["portfolio_id","annual_return","annual_volatility","sharpe_ratio","max_drawdown",
                 "sortino_ratio","calmar_ratio","skewness","kurtosis","beta","n_assets","hhi",
                 "pct_equity","pct_fixed_income","pct_commodities","pct_reits"]
    feat_cols = [c for c in feat_cols if c in feat_df.columns]
    feat_df[feat_cols].to_parquet(processed_dir / "portfolios_features.parquet", index=False)
    composition.to_parquet(processed_dir / "portfolios_composition.parquet", index=False)
    port_returns.to_parquet(processed_dir / "portfolios_returns.parquet")
    print("Saved: portfolios_features.parquet | portfolios_composition.parquet | portfolios_returns.parquet")
