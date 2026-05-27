"""
nb01_data.py — Data universe download, cleaning, and summary functions.
"""
import json
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

TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","JPM","JNJ","XOM","PG","NVDA","WMT",
    "VGK","EWJ","EWG","EWU","EEM","VWO",
    "TLT","IEF","SHY","BND","LQD","HYG",
    "GLD","SLV","USO","DBA",
    "VNQ","IYR",
    "XLK","XLF","XLE","XLV","XLI",
]
ASSET_CLASSES = {
    "AAPL":"US Equities","MSFT":"US Equities","GOOGL":"US Equities","AMZN":"US Equities",
    "JPM":"US Equities","JNJ":"US Equities","XOM":"US Equities","PG":"US Equities",
    "NVDA":"US Equities","WMT":"US Equities",
    "VGK":"International Equities","EWJ":"International Equities",
    "EWG":"International Equities","EWU":"International Equities",
    "EEM":"International Equities","VWO":"International Equities",
    "TLT":"Fixed Income","IEF":"Fixed Income","SHY":"Fixed Income",
    "BND":"Fixed Income","LQD":"Fixed Income","HYG":"Fixed Income",
    "GLD":"Commodities","SLV":"Commodities","USO":"Commodities","DBA":"Commodities",
    "VNQ":"REITs","IYR":"REITs",
    "XLK":"Sector ETFs","XLF":"Sector ETFs","XLE":"Sector ETFs",
    "XLV":"Sector ETFs","XLI":"Sector ETFs",
}
BROAD = {t: ("Equity" if "Equiti" in v or "Sector" in v else v)
         for t, v in ASSET_CLASSES.items()}

# ── v3 universe: 50 assets across 8 classes ──────────────────────────────────
TICKERS_V3 = [
    # US Large Cap (10, same as v2)
    "AAPL","MSFT","GOOGL","AMZN","JPM","JNJ","XOM","PG","NVDA","WMT",
    # US Broad Market & Factors (5 new) — adds size, value/growth, index
    "SPY","QQQ","IWM","VTV","VUG",
    # International Developed (6 v2 + 3 new) — Canada/Australia different sector mix
    "VGK","EWJ","EWG","EWU","EEM","VWO","EFA","EWC","EWA",
    # Emerging Market Countries (3 new) — Brazil/Taiwan/HK diverge from broad EM
    "EWZ","EWT","EWH",
    # Fixed Income (6 v2 + 2 new) — adds inflation-linked and broad agg
    "TLT","IEF","SHY","BND","LQD","HYG","TIP","AGG",
    # Commodities (4, same as v2)
    "GLD","SLV","USO","DBA",
    # REITs (2, same as v2)
    "VNQ","IYR",
    # Sector ETFs (5 v2 + 4 new) — adds defensive and cyclical differentiation
    "XLK","XLF","XLE","XLV","XLI","XLP","XLU","XLB","XLY",
]
ASSET_CLASSES_V3 = {
    **ASSET_CLASSES,
    "SPY":"US Equities","QQQ":"US Equities","IWM":"US Equities",
    "VTV":"US Equities","VUG":"US Equities",
    "EFA":"International Equities","EWC":"International Equities","EWA":"International Equities",
    "EWZ":"International Equities","EWT":"International Equities","EWH":"International Equities",
    "TIP":"Fixed Income","AGG":"Fixed Income",
    "XLP":"Sector ETFs","XLU":"Sector ETFs","XLB":"Sector ETFs","XLY":"Sector ETFs",
}
BROAD_V3 = {t: ("Equity" if "Equiti" in v or "Sector" in v else v)
            for t, v in ASSET_CLASSES_V3.items()}

MACRO_TICKERS = {"VIX": "^VIX", "S&P 500": "SPY", "10Y Treasury": "^TNX"}


def download_prices(
    tickers: List[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    import yfinance as yf
    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    raw.index = pd.to_datetime(raw.index)
    print(f"Raw download: {raw.shape}")
    return raw.sort_index()


def download_macro(start: str, end: str) -> pd.DataFrame:
    """Download VIX, S&P 500, and 10Y Treasury from Yahoo Finance."""
    import yfinance as yf
    frames = {}
    for name, sym in MACRO_TICKERS.items():
        try:
            d = yf.Ticker(sym).history(start=start, end=end, auto_adjust=True)["Close"]
            d.index = pd.to_datetime(d.index).tz_localize(None)
            frames[name] = d.rename(name)
            print(f"  {name}: {len(d)} rows")
        except Exception as e:
            print(f"  {name} failed: {e}")
    return pd.DataFrame(frames).sort_index()


def clean_prices(raw: pd.DataFrame, max_missing_pct: float = 0.10) -> pd.DataFrame:
    """Remove low-coverage tickers, remove extreme returns, keep only complete rows."""
    nan_pct = raw.isnull().mean()
    valid   = nan_pct[nan_pct < max_missing_pct].index.tolist()
    dropped = [t for t in raw.columns if t not in valid]
    if dropped:
        print(f"Dropped (>{max_missing_pct*100:.0f}% missing): {dropped}")

    prices = raw[valid].copy()

    log_ret = np.log(prices / prices.shift(1))
    anomaly = log_ret.abs() > 0.60
    prices[anomaly] = np.nan
    prices = prices.dropna(how="any")
    print(f"Clean prices: {prices.shape}  [{prices.index[0].date()} -> {prices.index[-1].date()}]")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns, dropping inf values."""
    r = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    print(f"Returns: {r.shape}")
    return r


def compute_summary_stats(returns: pd.DataFrame, td: int = 252) -> pd.DataFrame:
    """Annualized return, vol, Sharpe, max drawdown, skewness, kurtosis per asset."""
    ann_ret = returns.mean() * td
    ann_vol = returns.std() * np.sqrt(td)
    sharpe  = ann_ret / (ann_vol + 1e-9)
    cum     = (1 + returns).cumprod()
    max_dd  = (cum / cum.cummax() - 1).min()
    return pd.DataFrame({
        "ann_return":     ann_ret,
        "ann_volatility": ann_vol,
        "sharpe_ratio":   sharpe,
        "max_drawdown":   max_dd,
        "skewness":       returns.skew(),
        "kurtosis":       returns.kurtosis(),
    })


def build_asset_universe_meta(prices: pd.DataFrame) -> dict:
    """Build the asset_universe.json metadata dict."""
    valid_tickers = list(prices.columns)
    return {
        "tickers":          valid_tickers,
        "n_assets":         len(valid_tickers),
        "start_date":       str(prices.index[0].date()),
        "end_date":         str(prices.index[-1].date()),
        "n_trading_days":   len(prices),
        "asset_classes":    {t: ASSET_CLASSES.get(t, "Unknown") for t in valid_tickers},
        "broad_categories": {t: BROAD.get(t, "Unknown") for t in valid_tickers},
        "source":           "Yahoo Finance via yfinance (auto_adjust=True)",
        "version":          "v2 (2010-2024)",
    }


def plot_universe(prices: pd.DataFrame, stats: pd.DataFrame, fig_dir: Path) -> None:
    """Cumulative returns (log scale) + Sharpe distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors = {"Equity": "#1f77b4", "Fixed Income": "#2ca02c",
              "Commodities": "#ff7f0e", "REITs": "#9467bd"}
    norm = (prices / prices.iloc[0]) * 100
    for tk in prices.columns:
        cat = BROAD.get(tk, "Equity")
        axes[0].semilogy(norm.index, norm[tk], lw=0.6, alpha=0.5, color=colors.get(cat, "#999"))
    axes[0].set_title(f"Rentabilidad acumulada 2010-2024 ({len(prices.columns)} activos)")
    axes[0].set_ylabel("Índice (base=100, escala log)")
    axes[0].grid(alpha=0.3)

    axes[1].hist(stats["sharpe_ratio"].values, bins=20,
                 color="#1f77b4", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Sharpe Ratio Anualizado")
    axes[1].set_title("Distribución del Sharpe por activo")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = fig_dir / "01_universe_v2.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    return fig


def save_outputs(prices: pd.DataFrame, returns: pd.DataFrame,
                 macro: pd.DataFrame, universe: dict,
                 processed_dir: Path) -> None:
    """Persist all NB01 outputs."""
    prices.to_parquet(processed_dir / "prices_clean.parquet")
    returns.to_parquet(processed_dir / "returns_daily.parquet")
    returns.to_parquet(processed_dir / "log_returns.parquet")
    macro.to_parquet(processed_dir / "macro_data.parquet")
    with open(processed_dir / "asset_universe.json", "w") as f:
        json.dump(universe, f, indent=2)
    print(f"Saved prices_clean.parquet {prices.shape}")
    print(f"Saved returns_daily.parquet {returns.shape}")
    print(f"Saved macro_data.parquet {macro.shape}")
    print(f"Saved asset_universe.json ({universe['n_assets']} tickers)")
