"""
Alpha Vantage data collector — supports US + Indian (NSE/BSE) stocks.

Free tier limits (standard key):
  - 25 requests/day
  - 5 requests/minute
  - DAILY bars only (intraday requires a paid premium plan)

For intraday data, yfinance is used automatically (no API key required).
Alpha Vantage is used here for DAILY historical data which can supplement
the model with longer time horizons.

Symbol formats:
  US stocks  :  AAPL, TSLA
  NSE stocks :  RELIANCE.NS  → sent to API as  NSE:RELIANCE
  BSE stocks :  RELIANCE.BSE → sent to API as  BSE:RELIANCE
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

AV_BASE = "https://www.alphavantage.co/query"
# Simple in-process request throttle (5 req/min free tier)
_last_request_time: float = 0.0
_MIN_INTERVAL = 12.5  # seconds between requests  (60s / 5 = 12s, keep a small buffer)


def _av_symbol(ticker: str) -> str:
    """
    Convert yfinance-style ticker to Alpha Vantage symbol format.
      RELIANCE.NS  →  NSE:RELIANCE
      RELIANCE.BSE →  BSE:RELIANCE
      AAPL         →  AAPL  (unchanged)
    """
    if ticker.endswith(".NS"):
        return "NSE:" + ticker[:-3]
    if ticker.endswith(".BSE") or ticker.endswith(".BO"):
        return "BSE:" + ticker.rsplit(".", 1)[0]
    return ticker


def _throttle():
    """Ensure we don't exceed 5 requests/minute."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute standard technical indicators on an OHLCV dataframe."""
    close  = df["Close"]
    volume = df["Volume"]

    df["Returns_1h"]  = close.pct_change(1).fillna(0)
    df["Returns_4h"]  = close.pct_change(4).fillna(0)
    df["SMA_5"]       = close.rolling(5,  min_periods=1).mean()
    df["SMA_20"]      = close.rolling(20, min_periods=1).mean()
    df["EMA_12"]      = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"]      = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["Price_Momentum_5h"] = close.pct_change(5).fillna(0)

    # Bollinger Bands
    bb_std        = close.rolling(20, min_periods=1).std().fillna(0)
    df["BB_Upper"] = df["SMA_20"] + 2 * bb_std
    df["BB_Lower"] = df["SMA_20"] - 2 * bb_std

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta).clip(lower=0).rolling(14, min_periods=1).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # Volume features
    vol_ma  = volume.rolling(20, min_periods=1).mean()
    vol_std = volume.rolling(20, min_periods=1).std().fillna(1)
    df["Volume_MA"]              = vol_ma
    df["Abnormal_Volume_Zscore"] = ((volume - vol_ma) / (vol_std + 1e-9)).clip(-5, 5)
    df["Volatility_5h"]          = df["Returns_1h"].rolling(5, min_periods=1).std().fillna(0)
    df["Price_Change_Pct"]       = df["Returns_1h"]
    if "High" in df.columns and "Low" in df.columns:
        df["HL_Spread"] = (df["High"] - df["Low"]) / (close + 1e-9)

    return df.fillna(0).replace([np.inf, -np.inf], 0)


class AlphaVantageCollector:
    """
    Fetches intraday OHLCV data from Alpha Vantage.
    Supports US, NSE (India), and BSE (India) symbols.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY", "")
        if not self.api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY not set")

    def fetch_daily(
        self,
        ticker: str,
        outputsize: str = "compact",   # compact = last 100 days; full = 20+ years
    ) -> Optional[pd.DataFrame]:
        """
        Fetch DAILY OHLCV bars using the free Alpha Vantage endpoint.

        NOTE: TIME_SERIES_INTRADAY requires a premium plan.
              For intraday data, yfinance is used automatically.

        Args:
            ticker    : yfinance-style symbol (e.g. RELIANCE.NS, AAPL)
            outputsize: 'compact' (last 100 days) | 'full' (all history)

        Returns:
            DataFrame with daily OHLCV + technical indicators, or None on failure.
        """
        av_sym = _av_symbol(ticker)
        _throttle()

        params = {
            "function":   "TIME_SERIES_DAILY",
            "symbol":     av_sym,
            "outputsize": outputsize,
            "apikey":     self.api_key,
            "datatype":   "json",
        }

        try:
            resp = requests.get(AV_BASE, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if "Error Message" in data:
                logger.error(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
                return None
            if "Information" in data:
                logger.warning(f"Alpha Vantage premium required for {ticker}: {data['Information'][:100]}")
                return None
            if "Note" in data:
                logger.warning(f"Alpha Vantage rate-limit for {ticker}: {data['Note'][:100]}")
                return None

            key = "Time Series (Daily)"
            if key not in data:
                logger.warning(f"No daily time-series for {ticker}. Keys: {list(data.keys())}")
                return None

            ts = data[key]
            rows = []
            for dt_str, vals in ts.items():
                rows.append({
                    "timestamp": pd.to_datetime(dt_str),
                    "Open":   float(vals["1. open"]),
                    "High":   float(vals["2. high"]),
                    "Low":    float(vals["3. low"]),
                    "Close":  float(vals["4. close"]),
                    "Volume": float(vals["5. volume"]),
                })

            df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            df["ticker"] = ticker
            df = _build_features(df)
            logger.info(f"Alpha Vantage daily: {len(df)} bars for {ticker} ({av_sym})")
            return df

        except requests.RequestException as e:
            logger.error(f"Alpha Vantage request failed for {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Alpha Vantage processing error for {ticker}: {e}", exc_info=True)
            return None

    def fetch_latest_price(self, ticker: str) -> Optional[dict]:
        """
        Fetch the latest quote using the free GLOBAL_QUOTE endpoint.
        Returns dict with keys: price, open, high, low, volume, change_pct.
        """
        av_sym = _av_symbol(ticker)
        _throttle()
        params = {"function": "GLOBAL_QUOTE", "symbol": av_sym, "apikey": self.api_key}
        try:
            resp = requests.get(AV_BASE, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("Global Quote", {})
            if not data:
                return None
            return {
                "ticker":      ticker,
                "price":       float(data.get("05. price", 0)),
                "open":        float(data.get("02. open", 0)),
                "high":        float(data.get("03. high", 0)),
                "low":         float(data.get("04. low", 0)),
                "volume":      float(data.get("06. volume", 0)),
                "change_pct":  data.get("10. change percent", "0%"),
                "timestamp":   data.get("07. latest trading day"),
            }
        except Exception as e:
            logger.error(f"Alpha Vantage GLOBAL_QUOTE failed for {ticker}: {e}")
            return None

    def is_indian_ticker(self, ticker: str) -> bool:
        """Return True for NSE/BSE symbols."""
        return ticker.endswith(".NS") or ticker.endswith(".BSE") or ticker.endswith(".BO")
