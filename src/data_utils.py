"""
Data download, cleaning, and preprocessing utilities.

Functions for downloading market data via yfinance, cleaning price series,
computing returns, and downloading macro indicators.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ============================================================================
# ASSET UNIVERSE DEFINITION
# ============================================================================

ASSET_UNIVERSE: Dict[str, Dict[str, str]] = {
    "US Equities": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet",
        "AMZN": "Amazon",
        "JPM": "JPMorgan Chase",
        "JNJ": "Johnson & Johnson",
        "XOM": "Exxon Mobil",
        "PG": "Procter & Gamble",
        "NVDA": "NVIDIA",
        "WMT": "Walmart",
    },
    "International Equities": {
        "VGK": "Europe ETF",
        "EWJ": "Japan ETF",
        "EWG": "Germany ETF",
        "EWU": "UK ETF",
        "EEM": "Emerging Markets",
        "VWO": "Emerging Markets Vanguard",
    },
    "Fixed Income": {
        "TLT": "20+ Year Treasury",
        "IEF": "7-10 Year Treasury",
        "SHY": "1-3 Year Treasury",
        "BND": "Total Bond Market",
        "LQD": "Investment Grade Corporate",
        "HYG": "High Yield Corporate",
    },
    "Commodities": {
        "GLD": "Gold",
        "SLV": "Silver",
        "USO": "Oil",
        "DBA": "Agriculture",
    },
    "REITs": {
        "VNQ": "US REITs",
        "IYR": "US Real Estate",
    },
    "Sector ETFs": {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLI": "Industrials",
        "XLC": "Communication Services",
    },
}

# Asset class mapping for each ticker
ASSET_CLASS_MAP: Dict[str, str] = {}
for _cls, _tickers in ASSET_UNIVERSE.items():
    for _tkr in _tickers:
        ASSET_CLASS_MAP[_tkr] = _cls

# Broader category for portfolio allocation analysis
BROAD_CATEGORY_MAP: Dict[str, str] = {
    "US Equities": "Equity",
    "International Equities": "Equity",
    "Fixed Income": "Fixed Income",
    "Commodities": "Commodities",
    "REITs": "REITs",
    "Sector ETFs": "Equity",
}

MACRO_TICKERS: Dict[str, str] = {
    "^VIX": "VIX",
    "^GSPC": "S&P 500",
    "^TNX": "10Y Treasury Yield",
    "^IRX": "3M Treasury Yield",
}


def get_all_tickers() -> List[str]:
    """Return a flat list of all asset tickers in the universe."""
    tickers = []
    for assets in ASSET_UNIVERSE.values():
        tickers.extend(assets.keys())
    return tickers


def get_ticker_metadata() -> pd.DataFrame:
    """Return a DataFrame with ticker, name, asset_class, and broad_category."""
    rows = []
    for asset_class, assets in ASSET_UNIVERSE.items():
        for ticker, name in assets.items():
            rows.append({
                "ticker": ticker,
                "name": name,
                "asset_class": asset_class,
                "broad_category": BROAD_CATEGORY_MAP[asset_class],
            })
    return pd.DataFrame(rows)


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_prices(
    tickers: List[str],
    start: str = "2016-01-01",
    end: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Download adjusted close prices from yfinance.

    Args:
        tickers: List of ticker symbols.
        start: Start date string.
        end: End date string (defaults to today).

    Returns:
        Tuple of (prices DataFrame, successful tickers, failed tickers).
    """
    if end is None:
        end = pd.Timestamp.now().strftime("%Y-%m-%d")

    logger.info("Downloading %d tickers from %s to %s", len(tickers), start, end)

    successful = []
    failed = []
    all_prices = {}

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if data.empty or len(data) < 100:
                logger.warning("Ticker %s: insufficient data (%d rows)", ticker, len(data))
                failed.append(ticker)
                continue
            # Handle multi-level columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"].iloc[:, 0]
            else:
                close = data["Close"]
            all_prices[ticker] = close
            successful.append(ticker)
        except Exception as e:
            logger.warning("Ticker %s download failed: %s", ticker, e)
            failed.append(ticker)

    prices = pd.DataFrame(all_prices)
    prices.index = pd.to_datetime(prices.index)
    prices.index = prices.index.tz_localize(None)
    prices = prices.sort_index()

    logger.info("Downloaded %d/%d tickers successfully", len(successful), len(tickers))
    if failed:
        logger.warning("Failed tickers: %s", failed)

    return prices, successful, failed


def download_macro_data(
    start: str = "2016-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download macro/market indicators (VIX, S&P 500, Treasury yields).

    Returns:
        DataFrame with columns for each macro indicator.
    """
    if end is None:
        end = pd.Timestamp.now().strftime("%Y-%m-%d")

    macro_data = {}
    for ticker, name in MACRO_TICKERS.items():
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if data.empty:
                logger.warning("Macro ticker %s (%s): no data", ticker, name)
                continue
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"].iloc[:, 0]
            else:
                close = data["Close"]
            macro_data[name] = close
        except Exception as e:
            logger.warning("Macro ticker %s (%s) failed: %s", ticker, name, e)

    df = pd.DataFrame(macro_data)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    df = df.sort_index()
    return df


# ============================================================================
# DATA CLEANING
# ============================================================================

def clean_prices(
    prices: pd.DataFrame,
    max_nan_pct: float = 0.05,
    max_daily_return: float = 0.50,
) -> Tuple[pd.DataFrame, List[str]]:
    """Clean price data: remove tickers with excessive NaNs, forward-fill gaps.

    Args:
        prices: Raw price DataFrame.
        max_nan_pct: Maximum fraction of NaN allowed per ticker.
        max_daily_return: Maximum absolute daily return to flag anomalies.

    Returns:
        Tuple of (cleaned prices, list of removed tickers).
    """
    n_rows = len(prices)
    nan_pct = prices.isnull().sum() / n_rows
    removed = nan_pct[nan_pct > max_nan_pct].index.tolist()

    if removed:
        logger.info("Removing %d tickers with >%.0f%% NaN: %s",
                     len(removed), max_nan_pct * 100, removed)

    prices_clean = prices.drop(columns=removed).copy()

    # Forward-fill for minor gaps (holidays, etc.)
    prices_clean = prices_clean.ffill()

    # Drop any remaining rows with NaN (leading NaNs)
    prices_clean = prices_clean.dropna()

    # Check for anomalous returns (potential split issues)
    returns = prices_clean.pct_change().iloc[1:]
    anomalies = (returns.abs() > max_daily_return).any()
    if anomalies.any():
        bad_tickers = anomalies[anomalies].index.tolist()
        logger.warning("Tickers with daily returns >%.0f%%: %s", max_daily_return * 100, bad_tickers)

    return prices_clean, removed


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily logarithmic returns from prices.

    Args:
        prices: Clean price DataFrame.

    Returns:
        DataFrame of log returns (first row dropped).
    """
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.iloc[1:]


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple (arithmetic) returns from prices.

    Args:
        prices: Clean price DataFrame.

    Returns:
        DataFrame of simple returns (first row dropped).
    """
    return prices.pct_change().iloc[1:]
