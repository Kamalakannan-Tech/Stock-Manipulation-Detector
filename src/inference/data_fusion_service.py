"""
Data fusion service that combines real-time market and social data for model inference.
Data source priority:
  1. MongoDB (live Alpaca stream data)
  2. Alpha Vantage (Indian NSE/BSE tickers)
  3. yfinance (US + Indian fallback, ~15-min delayed)
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
try:
    from pymongo import MongoClient
    _PYMONGO_AVAILABLE = True
except ImportError:
    MongoClient = None          # type: ignore
    _PYMONGO_AVAILABLE = False
import json
import os
import sys

# Import cache fallback
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.cache_fallback import get_redis_client

# Alpha Vantage collector (optional — only used when key is set)
try:
    from src.data_collection.alpha_vantage_collector import AlphaVantageCollector
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False

# Finnhub collector (optional — only used when key is set)
try:
    from src.data_collection.finnhub_collector import FinnhubCollector
    _FH_AVAILABLE = True
except ImportError:
    _FH_AVAILABLE = False

# StockTwits collector (no API key required — always available)
try:
    from src.data_collection.stocktwits_collector import StockTwitsCollector
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

logger = logging.getLogger(__name__)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _fetch_yfinance_data(ticker: str, period: str = '1d', interval: str = '1m') -> Optional[pd.DataFrame]:
    """Fetch real-time market data from yfinance as fallback."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return None
        df = df.reset_index()
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.columns = [str(c).strip() for c in df.columns]
        # Remove duplicates explicitly to prevent duplicate 'Close' columns
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        return df
    except Exception as e:
        logger.error(f"yfinance fetch failed for {ticker}: {e}")
        return None


def _build_features_from_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators from OHLCV data."""
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated(keep='first')] # Ensure no exact dupes
    close = df['Close'].astype(float)
    volume = df['Volume'].astype(float)

    # Use fill_method=None to avoid pandas FutureWarnings
    df['return_1tick'] = close.pct_change(fill_method=None)
    df['sma_5'] = close.rolling(5, min_periods=1).mean()
    df['sma_20'] = close.rolling(20, min_periods=1).mean()
    df['ema_12'] = close.ewm(span=12, adjust=False).mean()
    df['volatility_10tick'] = df['return_1tick'].rolling(10, min_periods=1).std()

    # RSI
    df['rsi'] = _compute_rsi(close)

    # MACD
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - ema26

    # Bollinger Bands
    rolling_std = close.rolling(20, min_periods=1).std()
    df['bb_upper'] = df['sma_20'] + 2 * rolling_std
    df['bb_lower'] = df['sma_20'] - 2 * rolling_std

    # Volume z-score
    vol_mean = volume.rolling(20, min_periods=1).mean()
    vol_std = volume.rolling(20, min_periods=1).std().fillna(1)
    df['volume_zscore'] = (volume - vol_mean) / (vol_std + 1e-9)
    df['volume_ratio'] = volume / (vol_mean + 1e-9)

    # Price momentum
    df['price_mom5'] = close.pct_change(5)

    # HL spread
    if 'High' in df.columns and 'Low' in df.columns:
        df['hl_spread'] = (df['High'] - df['Low']) / (close + 1e-9)
    else:
        df['hl_spread'] = 0.0

    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    return df


class DataFusionService:
    """
    Fuses real-time market data and social media data for model inference.
    Falls back to yfinance when MongoDB has no live streaming data.
    """

    def __init__(self,
                 mongo_uri='mongodb://localhost:27017/',
                 mongo_db='stock_manipulation',
                 redis_url='redis://localhost:6379',
                 market_window_minutes=60,
                 social_window_hours=24):
        # Database connections
        try:
            self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
            self.mongo_client.admin.command('ping')
            self.db = self.mongo_client[mongo_db]
            self.market_collection = self.db['live_market_data']
            self.social_collection = self.db['live_social_data']
            self.sentiment_collection = self.db['social_sentiment']
            self.mongo_available = True
            logger.info("MongoDB connected")
        except Exception as e:
            logger.warning(f"MongoDB unavailable ({e}), will use yfinance only")
            self.mongo_available = False
            self.db = None
            self.market_collection = None
            self.social_collection = None
            self.sentiment_collection = None

        # Redis/cache with fallback
        self.redis_client = get_redis_client(redis_url, decode_responses=True)

        self.market_window = timedelta(minutes=market_window_minutes)
        self.social_window = timedelta(hours=social_window_hours)

        # Alpha Vantage collector (initialised lazily)
        self._av: Optional[AlphaVantageCollector] = None
        av_key = os.getenv('ALPHAVANTAGE_API_KEY', '')
        if _AV_AVAILABLE and av_key:
            try:
                self._av = AlphaVantageCollector(api_key=av_key)
                logger.info('Alpha Vantage collector ready')
            except Exception as e:
                logger.warning(f'Alpha Vantage init failed: {e}')

        # Finnhub collector (real news sentiment for US stocks)
        self._fh: Optional[FinnhubCollector] = None
        fh_key = os.getenv('FINNHUB_API_KEY', '')
        if _FH_AVAILABLE and fh_key:
            try:
                self._fh = FinnhubCollector(api_key=fh_key)
                logger.info('Finnhub collector ready (real news sentiment)')
            except Exception as e:
                logger.warning(f'Finnhub init failed: {e}')

        logger.info("Data fusion service initialized")

    def get_market_features(self, ticker: str, lookback_minutes: int = 60) -> Optional[pd.DataFrame]:
        """
        Get recent market data features for a ticker.
        Tries MongoDB first, falls back to yfinance.
        """
        # 1. Try MongoDB (live stream data)
        if self.mongo_available and self.market_collection is not None:
            try:
                cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
                cursor = self.market_collection.find({
                    'ticker': ticker,
                    'timestamp': {'$gte': cutoff_time},
                    'data_type': 'trade'
                }).sort('timestamp', 1)
                data = list(cursor)

                if data:
                    df = pd.DataFrame(data)
                    required_cols = ['timestamp', 'price', 'size']
                    if all(col in df.columns for col in required_cols):
                        df = df.sort_values('timestamp')
                        if 'return_1tick' not in df.columns and len(df) > 1:
                            df['return_1tick'] = df['price'].pct_change()
                        if 'sma_5' not in df.columns and len(df) >= 5:
                            df['sma_5'] = df['price'].rolling(5, min_periods=1).mean()
                        if 'volatility_10tick' not in df.columns and len(df) >= 10:
                            df['volatility_10tick'] = df['return_1tick'].rolling(10, min_periods=1).std()
                        logger.info(f"MongoDB: {len(df)} market points for {ticker}")
                        return df
            except Exception as e:
                logger.warning(f"MongoDB market query failed: {e}")

        # 2. yfinance — works for both US and Indian (.NS/.BSE) tickers, ~15 min delayed
        logger.info(f"Fetching {ticker} from yfinance (1-minute bars, last 5 days)...")
        df = _fetch_yfinance_data(ticker, period='5d', interval='1m')
        if df is None or df.empty:
            logger.warning(f"No market data available for {ticker}")
            return None

        # Rename columns to standard names
        col_map = {'Datetime': 'timestamp', 'Open': 'Open', 'High': 'High',
                   'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'timestamp'})
        elif 'Date' in df.columns:
            df = df.rename(columns={'Date': 'timestamp'})

        # Keep only last lookback_minutes rows
        df = df.tail(lookback_minutes).reset_index(drop=True)

        # Build technical features
        df = _build_features_from_ohlcv(df)

        # Add 'price' column (alias for Close)
        df['price'] = df['Close']
        df['size'] = df['Volume']
        df['ticker'] = ticker

        logger.info(f"yfinance: {len(df)} market points for {ticker}")
        return df

    def get_social_features(self, ticker: str, lookback_hours: int = 24) -> Dict:
        """
        Get aggregated social media features for a ticker.
        Returns default neutral features when no social data is available.
        """
        # Check Redis cache
        redis_key = f"social:sentiment:{ticker}"
        cached_data = self.redis_client.get(redis_key)
        if cached_data:
            try:
                return json.loads(cached_data)
            except Exception:
                pass

        # Try MongoDB
        if self.mongo_available and self.sentiment_collection is not None:
            try:
                cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
                cursor = self.sentiment_collection.find({
                    'ticker': ticker,
                    'timestamp': {'$gte': cutoff_time}
                }).sort('timestamp', -1).limit(1)
                data = list(cursor)

                if data:
                    latest = data[0]
                    features = {
                        'overall_sentiment': latest.get('overall_sentiment', 0),
                        'total_posts': latest.get('total_posts', 0),
                        'total_engagement': latest.get('total_engagement', 0),
                        'post_velocity': latest.get('post_velocity', 0),
                        'timestamp': latest.get('timestamp', datetime.now())
                    }
                    if 'sentiment_by_source' in latest:
                        for source, stats in latest['sentiment_by_source'].items():
                            features[f'{source}_posts'] = stats.get('posts', 0)
                            features[f'{source}_sentiment'] = stats.get('sentiment', 0)
                            features[f'{source}_engagement'] = stats.get('engagement', 0)
                    return features
            except Exception as e:
                logger.warning(f"MongoDB social query failed: {e}")

        # 1b. Finnhub real sentiment (US stocks only, before MongoDB)
        is_us = '.' not in ticker
        if self._fh is not None and is_us:
            try:
                fh_features = self._fh.build_social_features(ticker)
                if fh_features and any(v != 0.0 for v in fh_features):
                    logger.info(f'Finnhub real sentiment for {ticker}')
                    # Cache it for 15 minutes
                    self.redis_client.setex(
                        f'social:{ticker}', 900,
                        json.dumps({'_finnhub': fh_features})
                    )
                    return {'_finnhub': fh_features}  # consumed in prepare_model_input
            except Exception as e:
                logger.warning(f'Finnhub social fetch failed for {ticker}: {e}')

        logger.warning(f"No social data for {ticker}, using neutral defaults")
        return self._get_default_social_features()

    def _get_default_social_features(self) -> Dict:
        """Return default neutral social features."""
        return {
            'overall_sentiment': 0,
            'total_posts': 0,
            'total_engagement': 0,
            'post_velocity': 0,
            'twitter_posts': 0,
            'twitter_sentiment': 0,
            'twitter_engagement': 0,
            'reddit_posts': 0,
            'reddit_sentiment': 0,
            'reddit_engagement': 0,
            'stocktwits_posts': 0,
            'stocktwits_sentiment': 0,
            'stocktwits_engagement': 0,
            'timestamp': datetime.now()
        }

    def prepare_model_input(self, ticker: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare fused data for model inference.
        Returns (social_features, market_features) as numpy arrays shaped for the model.
        """
        try:
            market_df = self.get_market_features(ticker, lookback_minutes=60)
            if market_df is None or len(market_df) < 5:
                logger.warning(f"Insufficient market data for {ticker}")
                return None

            social_dict = self.get_social_features(ticker, lookback_hours=24)

            # Social feature vector (6 features)
            # If Finnhub returned a raw 6-element list, use it directly
            if isinstance(social_dict, dict) and '_finnhub' in social_dict:
                social_vec = np.array(social_dict['_finnhub'], dtype=np.float32)
            else:
                social_vec = np.array([
                    float(social_dict.get('overall_sentiment', 0)),
                    float(social_dict.get('total_posts', 0)),
                    float(social_dict.get('total_engagement', 0)),
                    float(social_dict.get('post_velocity', 0)),
                    float(social_dict.get('twitter_sentiment', 0)),
                    float(social_dict.get('reddit_sentiment', 0))
                ], dtype=np.float32)

            # Market feature columns (13 features matching training)
            market_feature_cols = [
                'Close', 'Volume', 'return_1tick', 'sma_5', 'sma_20',
                'volatility_10tick', 'volume_ratio', 'rsi', 'macd',
                'bb_upper', 'bb_lower', 'volume_zscore', 'hl_spread'
            ]

            # Use available columns, pad if needed
            available_cols = [c for c in market_feature_cols if c in market_df.columns]
            if len(available_cols) < 3:
                # Try alternate column names
                alt_cols = ['price', 'size', 'return_1tick', 'sma_5', 'sma_20', 'volatility_10tick']
                available_cols = [c for c in alt_cols if c in market_df.columns]

            if not available_cols:
                logger.warning(f"No usable market columns for {ticker}")
                return None

            sequence_length = 30
            market_arr = market_df[available_cols].tail(sequence_length).values.astype(np.float32)

            # Pad columns to 13
            if market_arr.shape[1] < 13:
                pad = np.zeros((market_arr.shape[0], 13 - market_arr.shape[1]), dtype=np.float32)
                market_arr = np.hstack([market_arr, pad])
            elif market_arr.shape[1] > 13:
                market_arr = market_arr[:, :13]

            # Pad rows to sequence_length
            if len(market_arr) < sequence_length:
                pad_rows = np.zeros((sequence_length - len(market_arr), 13), dtype=np.float32)
                market_arr = np.vstack([pad_rows, market_arr])

            # Replace NaN/Inf
            market_arr = np.nan_to_num(market_arr, nan=0.0, posinf=0.0, neginf=0.0)

            # Expand social to match sequence length
            social_expanded = np.tile(social_vec, (sequence_length, 1))

            logger.info(f"Model input ready for {ticker}: social={social_expanded.shape}, market={market_arr.shape}")
            return social_expanded, market_arr

        except Exception as e:
            logger.error(f"Error preparing model input for {ticker}: {e}", exc_info=True)
            return None

    def get_latest_data_summary(self, ticker: str) -> Dict:
        """Get a summary of latest data for a ticker."""
        summary = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'market_data_available': False,
            'social_data_available': False
        }

        market_df = self.get_market_features(ticker, lookback_minutes=60)
        if market_df is not None and not market_df.empty:
            summary['market_data_available'] = True
            price_col = 'Close' if 'Close' in market_df.columns else 'price'
            if price_col in market_df.columns:
                summary['latest_price'] = float(market_df[price_col].iloc[-1])
            summary['market_data_points'] = len(market_df)
            ts_col = 'timestamp' if 'timestamp' in market_df.columns else None
            if ts_col:
                summary['latest_market_timestamp'] = market_df[ts_col].iloc[-1]

        social_dict = self.get_social_features(ticker, lookback_hours=24)
        if social_dict and social_dict.get('total_posts', 0) > 0:
            summary['social_data_available'] = True
            summary['social_sentiment'] = social_dict.get('overall_sentiment', 0)
            summary['social_posts'] = social_dict.get('total_posts', 0)
            summary['latest_social_timestamp'] = social_dict.get('timestamp')

        return summary

    def close(self):
        """Close database connections."""
        if self.mongo_available and self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            self.redis_client.close()
