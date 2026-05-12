"""
Sector Decoupling Module
Computes stock-specific abnormal returns by removing sector and market-wide movements.
"""
import numpy as np
import pandas as pd
import logging
import yfinance as yf
from pathlib import Path

logger = logging.getLogger(__name__)

# Sector ETF proxies for each stock
SECTOR_MAP = {
    'GME':  {'sector_etf': 'XLY',  'peers': ['AMC', 'BBBY']},
    'AMC':  {'sector_etf': 'XLY',  'peers': ['GME']},
    'TSLA': {'sector_etf': 'XLY',  'peers': ['RIVN', 'F', 'GM']},
    'AAPL': {'sector_etf': 'XLK',  'peers': ['MSFT', 'GOOGL']},
    'MSFT': {'sector_etf': 'XLK',  'peers': ['AAPL', 'GOOGL']},
    'NVDA': {'sector_etf': 'SOXX', 'peers': ['AMD', 'INTC']},
}

MARKET_TICKER = 'SPY'


class SectorDecoupler:
    """
    Removes sector and market-wide movements to isolate stock-specific signals.
    
    Abnormal return = stock_return - (0.5 * sector_return + 0.3 * market_return)
    """

    def __init__(self, market_weight: float = 0.3, sector_weight: float = 0.5,
                 zscore_threshold: float = 2.0):
        self.market_weight = market_weight
        self.sector_weight = sector_weight
        self.zscore_threshold = zscore_threshold
        self._cache: dict = {}

    def _load_returns(self, ticker: str, start: str, end: str, interval: str = '1h') -> pd.Series:
        """Load periodic returns for a ticker from yfinance or local cache."""
        # Check local raw data first
        local_path = Path(f'data/raw/market/{ticker}_market.csv')
        if local_path.exists():
            try:
                df = pd.read_csv(local_path)
                ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
                df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
                df = df.set_index(ts_col).sort_index()
                return df['Close'].pct_change().fillna(0)
            except Exception:
                pass
        # Fallback: fetch live
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval,
                             progress=False, auto_adjust=True)
            return df['Close'].pct_change().fillna(0)
        except Exception as e:
            logger.warning(f"Could not load returns for {ticker}: {e}")
            return pd.Series(dtype=float)

    def compute_abnormal_returns(self, ticker: str, returns: pd.Series,
                                  index: pd.DatetimeIndex | None = None) -> pd.Series:
        """
        Compute abnormal returns for a stock.
        
        Args:
            ticker: Stock symbol
            returns: Series of periodic returns for the stock
            index: DatetimeIndex to align on (uses returns.index if None)
        
        Returns:
            Series of abnormal returns
        """
        if index is None:
            index = returns.index

        start = str(index.min().date()) if hasattr(index.min(), 'date') else '2025-01-01'
        end   = str(index.max().date()) if hasattr(index.max(), 'date') else '2026-03-01'

        # Market returns
        market_returns = self._get_or_fetch(MARKET_TICKER, start, end)
        market_aligned = self._align(market_returns, returns)

        # Sector returns
        sector_etf = SECTOR_MAP.get(ticker, {}).get('sector_etf', 'SPY')
        sector_returns = self._get_or_fetch(sector_etf, start, end)
        sector_aligned = self._align(sector_returns, returns)

        abnormal = returns.values - (
            self.sector_weight * sector_aligned +
            self.market_weight  * market_aligned
        )
        return pd.Series(abnormal, index=returns.index, name='abnormal_return')

    def compute_zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Rolling z-score normalisation."""
        mean = series.rolling(window, min_periods=5).mean()
        std  = series.rolling(window, min_periods=5).std().fillna(1e-9).replace(0, 1e-9)
        return ((series - mean) / std).clip(-10, 10).fillna(0)

    def should_alert(self, zscore: float) -> bool:
        return abs(zscore) >= self.zscore_threshold

    def enrich_dataframe(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Add `abnormal_return`, `abnormal_zscore`, and `sector_divergence`
        columns to an existing market DataFrame.
        """
        if 'Returns_1h' not in df.columns:
            df['Returns_1h'] = df['Close'].pct_change().fillna(0)
        returns = df['Returns_1h']

        try:
            abnormal = self.compute_abnormal_returns(ticker, returns, index=returns.index)
            df['abnormal_return']  = abnormal.values
            df['abnormal_zscore']  = self.compute_zscore(abnormal).values
            df['sector_divergence'] = df['abnormal_return']
        except Exception as e:
            logger.warning(f"Sector decoupling failed for {ticker}: {e}. Using raw returns.")
            df['abnormal_return']  = returns
            df['abnormal_zscore']  = self.compute_zscore(returns).values
            df['sector_divergence'] = returns
        return df

    def _get_or_fetch(self, ticker: str, start: str, end: str) -> pd.Series:
        if ticker not in self._cache:
            self._cache[ticker] = self._load_returns(ticker, start, end)
        return self._cache[ticker]

    @staticmethod
    def _align(ref: pd.Series, target: pd.Series) -> np.ndarray:
        """Align ref series to target's length, returning a 1-D float array."""
        n = len(target)
        if ref.empty:
            return np.zeros(n)
        try:
            # Use integer position instead of index-label alignment to avoid
            # the (N, N) matrix produced when the two indices don't match.
            ref_arr = np.asarray(ref).astype(float).ravel()
            if len(ref_arr) >= n:
                return ref_arr[:n]
            # Repeat last value to pad
            return np.pad(ref_arr, (0, n - len(ref_arr)), mode='edge')
        except Exception:
            return np.zeros(n)

