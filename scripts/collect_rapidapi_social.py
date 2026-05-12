"""
rapidapi_social_collector.py
============================
Collects Reddit + StockTwits social sentiment data via RapidAPI for all
monitored tickers and saves them as {ticker}_social.csv in data/raw/social/.

RapidAPI services used:
  - Reddit:     "Reddit Search API" (host: reddit-scraper2.p.rapidapi.com)
                Free tier: 100 requests/month
  - StockTwits: "Finance Social Sentiment" (host: finance-social-sentiment-for-twitter-and-stocktwits.p.rapidapi.com)
                Free tier: 50 requests/day

SETUP (one-time):
  1. Go to https://rapidapi.com/ and create a free account
  2. Subscribe to both APIs above (both have free tiers)
  3. Copy your RapidAPI key from the Dashboard
  4. Add to your .env file:
       RAPIDAPI_KEY=your_rapidapi_key_here

OUTPUT:
  data/raw/social/{ticker}_social.csv  with columns:
    timestamp, ticker, sentiment_score, post_volume, engagement_score,
    bot_activity_score, narrative_similarity, sector_divergence

Usage:
    python scripts/collect_rapidapi_social.py
    python scripts/collect_rapidapi_social.py --tickers GME AMC TSLA
    python scripts/collect_rapidapi_social.py --mode reddit-only
    python scripts/collect_rapidapi_social.py --mode stocktwits-only
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, '.')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SOCIAL_DIR = Path('data/raw/social')
SOCIAL_DIR.mkdir(parents=True, exist_ok=True)

RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY', '')

SUBREDDITS = ['wallstreetbets', 'stocks', 'StockMarket', 'investing', 'pennystocks']
POSTS_PER_SUBREDDIT = 50


# ─────────────────────────────────────────────────────────────────────────────
# Reddit via RapidAPI
# ─────────────────────────────────────────────────────────────────────────────

class RedditRapidAPI:
    """
    Uses 'Reddit' API on RapidAPI by social miner.
    Host: reddit34.p.rapidapi.com
    Docs: https://rapidapi.com/social-miner/api/reddit
    """

    HOST = "reddit34.p.rapidapi.com"

    def __init__(self, api_key: str):
        self.headers = {
            "x-rapidapi-host": self.HOST,
            "x-rapidapi-key":  api_key,
        }

    def search_posts(self, ticker: str, subreddit: str = None, limit: int = 50) -> list[dict]:
        """Search Reddit posts mentioning a ticker."""
        query = f"${ticker}" if not ticker.startswith('$') else ticker

        if subreddit:
            url = f"https://{self.HOST}/getPostsBySubreddit"
            params = {"subreddit": subreddit, "sort": "new"}
        else:
            url = f"https://{self.HOST}/getSearchPosts"
            params = {"query": query, "sort": "new", "time": "month"}

        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            if resp.status_code == 429:
                logger.warning("Reddit RapidAPI rate limit hit — waiting 30s")
                time.sleep(30)
                resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            # API returns {posts: [...]} or a list directly
            posts = data.get('posts', data) if isinstance(data, dict) else data
            if isinstance(posts, dict):
                posts = posts.get('data', [])
            logger.info(f"  Reddit [{subreddit or 'all'}] ${ticker}: {len(posts)} posts")
            return posts[:limit]
        except Exception as e:
            logger.error(f"  Reddit API error for ${ticker}: {e}")
            return []

    def collect_ticker(self, ticker: str) -> pd.DataFrame:
        """Collect posts from broad search for a ticker (subreddit-specific calls skipped
        as /getPostsBySubreddit returns a different response shape)."""
        all_posts = []

        # Broad search — reliable, returns ~50 posts per query
        for query in [f'${ticker}', ticker]:
            posts = self.search_posts(ticker, limit=POSTS_PER_SUBREDDIT)
            all_posts.extend(posts)
            time.sleep(2)
            if len(all_posts) >= 50:
                break

        if not all_posts:
            return pd.DataFrame()

        rows = []
        for p in all_posts:
            try:
                rows.append({
                    'timestamp':     p.get('created_utc', datetime.now(timezone.utc).timestamp()),
                    'title':         p.get('title', ''),
                    'score':         int(p.get('score', 0)),
                    'num_comments':  int(p.get('num_comments', 0)),
                    'upvote_ratio':  float(p.get('upvote_ratio', 0.5)),
                    'subreddit':     p.get('subreddit', ''),
                    'author':        p.get('author', ''),
                })
            except Exception:
                continue

        df = pd.DataFrame(rows).drop_duplicates()
        if df.empty:
            return df

        # Convert unix timestamp
        if df['timestamp'].dtype in [np.float64, np.int64]:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

        df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        logger.info(f"  Reddit: collected {len(df)} unique posts for ${ticker}")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# StockTwits via RapidAPI — Finance Social Sentiment
# ─────────────────────────────────────────────────────────────────────────────

class StockTwitsRapidAPI:
    """
    Uses 'Finance Social Sentiment For Twitter and StockTwits' on RapidAPI.
    Host: finance-social-sentiment-for-twitter-and-stocktwits.p.rapidapi.com
    Docs: https://rapidapi.com/UtradeaAPI/api/finance-social-sentiment-for-twitter-and-stocktwits
    """

    HOST = "finance-social-sentiment-for-twitter-and-stocktwits.p.rapidapi.com"

    def __init__(self, api_key: str):
        self.headers = {
            "x-rapidapi-host": self.HOST,
            "x-rapidapi-key":  api_key,
        }

    def get_sentiment(self, ticker: str) -> dict | None:
        """Get social feed / sentiment data for a ticker."""
        url = f"https://{self.HOST}/get-social-feed"
        params = {"social": "stocktwits", "tickers": ticker, "timestamp": "24h"}
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            if resp.status_code == 429:
                logger.warning("StockTwits RapidAPI rate limit hit — waiting 60s")
                time.sleep(60)
                resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"  StockTwits sentiment retrieved for ${ticker}")
            return data
        except Exception as e:
            logger.error(f"  StockTwits API error for ${ticker}: {e}")
            return None

    def get_sentiment_history(self, ticker: str) -> pd.DataFrame | None:
        """Get 72h sentiment timeseries for a ticker."""
        url = f"https://{self.HOST}/get-social-feed"
        params = {"social": "stocktwits", "tickers": ticker, "timestamp": "72h"}
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            # Normalise various response shapes
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                inner = data.get(ticker, data.get('data', data.get('feed', [])))
                df = pd.DataFrame(inner) if isinstance(inner, list) else pd.DataFrame([data])
            else:
                return None
            logger.info(f"  StockTwits history: {len(df)} rows for ${ticker}")
            return df
        except Exception as e:
            logger.warning(f"  StockTwits history unavailable for ${ticker}: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering: raw API data → training feature format
# ─────────────────────────────────────────────────────────────────────────────

def reddit_to_features(reddit_df: pd.DataFrame, ticker: str,
                        target_timestamps: pd.DatetimeIndex = None) -> pd.DataFrame:
    """
    Convert raw Reddit posts to hourly social features aligned to market data.

    Features produced:
      - sentiment_score       : bullish/bearish keyword ratio  (-1 to 1)
      - post_volume           : posts per hour
      - engagement_score      : upvotes + comments*2
      - bot_activity_score    : low upvote_ratio → more bots (0 to 1)
      - narrative_similarity  : consensus of sentiment direction (0 to 1)
      - sector_divergence     : placeholder (set to 0 — filled by SectorDecoupler)
    """
    BULLISH = ['moon', 'rocket', 'buy', 'calls', 'bullish', 'squeeze', 'pump',
               'long', 'calls', 'yolo', 'tendies', 'ath', 'breakout']
    BEARISH = ['crash', 'dump', 'puts', 'bearish', 'short', 'sell', 'overvalued',
               'bubble', 'dead', 'rekt', 'bag', 'loss']

    df = reddit_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp'])

    title = df['title'].fillna('').str.lower()
    df['bullish'] = title.apply(lambda t: sum(1 for w in BULLISH if w in t))
    df['bearish'] = title.apply(lambda t: sum(1 for w in BEARISH if w in t))
    df['raw_sent'] = (df['bullish'] - df['bearish']).clip(-3, 3) / 3.0

    # Resample to hourly bins
    df = df.set_index('timestamp').sort_index()
    hourly = df.resample('1h').agg({
        'raw_sent':      'mean',
        'score':         'sum',
        'num_comments':  'sum',
        'upvote_ratio':  'mean',
        'bullish':       'sum',
        'bearish':       'sum',
    }).fillna(0)

    hourly['post_volume']         = df.resample('1h').size().fillna(0)
    hourly['engagement_score']    = hourly['score'] + hourly['num_comments'] * 2
    hourly['sentiment_score']     = hourly['raw_sent'].clip(-1, 1)
    hourly['bot_activity_score']  = (1 - hourly['upvote_ratio']).clip(0, 1)
    total_mentions = hourly['bullish'] + hourly['bearish'] + 1e-9
    hourly['narrative_similarity'] = (
        (hourly['bullish'].abs() / total_mentions).clip(0, 1) * 0.7
        + (1 - hourly['bot_activity_score']) * 0.3
    )
    hourly['sector_divergence'] = 0.0
    hourly['ticker']            = ticker

    result = hourly[['ticker', 'sentiment_score', 'post_volume',
                      'engagement_score', 'bot_activity_score',
                      'narrative_similarity', 'sector_divergence']].reset_index()

    if target_timestamps is not None:
        result = result.set_index('timestamp').reindex(
            target_timestamps, method='nearest', tolerance='1h'
        ).fillna(0).reset_index().rename(columns={'index': 'timestamp'})

    logger.info(f"  Reddit features: {len(result)} hourly rows for ${ticker}")
    return result


def stocktwits_to_features(st_data: dict | pd.DataFrame, ticker: str,
                            n_rows: int = None) -> pd.DataFrame | None:
    """
    Convert StockTwits API response to social feature format.
    Handles both the current snapshot and historical timeseries formats.
    """
    try:
        if isinstance(st_data, pd.DataFrame) and not st_data.empty:
            df = st_data.copy()
        elif isinstance(st_data, dict):
            # Current snapshot: create a single-row dataframe
            df = pd.DataFrame([st_data])
        else:
            return None

        # Normalize column names (API returns varying names)
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]

        row_map = {
            'sentiment_score':      df.get('sentiment', df.get('bullish_percent', None)),
            'post_volume':          df.get('posts', df.get('messages', df.get('volume', None))),
            'engagement_score':     df.get('engagement', df.get('impressions', None)),
        }

        result = pd.DataFrame()
        for col, series in row_map.items():
            if series is not None:
                result[col] = series.values if hasattr(series, 'values') else [series]

        if result.empty:
            logger.warning(f"  StockTwits: could not parse response for ${ticker}")
            return None

        # Fill missing features
        if 'sentiment_score' in result.columns:
            # StockTwits bullish_percent is 0-100; normalize to -1 to 1
            result['sentiment_score'] = (result['sentiment_score'].fillna(50) / 50 - 1).clip(-1, 1)
        if 'post_volume' not in result.columns:
            result['post_volume'] = 0
        if 'engagement_score' not in result.columns:
            result['engagement_score'] = result.get('post_volume', 0) * 3

        result['bot_activity_score']  = 0.1   # StockTwits has moderation; assume low bots
        result['narrative_similarity'] = result['sentiment_score'].abs().clip(0, 1)
        result['sector_divergence']   = 0.0
        result['ticker']              = ticker

        # If n_rows specified, broadcast/repeat to match market data length
        if n_rows and len(result) < n_rows:
            result = pd.concat([result] * (n_rows // len(result) + 1),
                               ignore_index=True).iloc[:n_rows]

        logger.info(f"  StockTwits features: {len(result)} rows for ${ticker}")
        return result

    except Exception as e:
        logger.error(f"  StockTwits feature conversion error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main collection pipeline
# ─────────────────────────────────────────────────────────────────────────────

def collect_ticker(ticker: str, reddit: RedditRapidAPI, stocktwits: StockTwitsRapidAPI,
                   mode: str = 'both') -> pd.DataFrame | None:
    """Collect and merge Reddit + StockTwits features for one ticker."""

    reddit_features     = None
    stocktwits_features = None

    if mode in ('both', 'reddit-only'):
        logger.info(f"\n[{ticker}] Collecting Reddit data...")
        reddit_df = reddit.collect_ticker(ticker)
        if not reddit_df.empty:
            reddit_features = reddit_to_features(reddit_df, ticker)

    if mode in ('both', 'stocktwits-only'):
        logger.info(f"[{ticker}] Collecting StockTwits data...")
        # Try historical first, fall back to current snapshot
        st_hist = stocktwits.get_sentiment_history(ticker)
        if st_hist is not None and not st_hist.empty:
            stocktwits_features = stocktwits_to_features(st_hist, ticker)
        else:
            st_now = stocktwits.get_sentiment(ticker)
            if st_now:
                stocktwits_features = stocktwits_to_features(st_now, ticker)
        time.sleep(1)

    # Merge available features
    if reddit_features is not None and stocktwits_features is not None:
        # Blend: average sentiment from both sources, take max engagement
        n = max(len(reddit_features), len(stocktwits_features))
        rf = reddit_features.iloc[:n].reset_index(drop=True)
        sf = stocktwits_features.iloc[:n].reset_index(drop=True)
        merged = rf.copy()
        for col in ['sentiment_score', 'narrative_similarity']:
            if col in sf.columns:
                merged[col] = (merged.get(col, 0) + sf[col].values) / 2
        for col in ['post_volume', 'engagement_score']:
            if col in sf.columns:
                merged[col] = merged.get(col, 0) + sf[col].fillna(0).values
        merged['bot_activity_score'] = sf.get('bot_activity_score', merged.get('bot_activity_score', 0.1))
        result = merged
    elif reddit_features is not None:
        result = reddit_features
    elif stocktwits_features is not None:
        result = stocktwits_features
    else:
        logger.warning(f"[{ticker}] No data collected from any source")
        return None

    # Ensure all required columns present
    REQ = ['timestamp', 'ticker', 'sentiment_score', 'post_volume',
           'engagement_score', 'bot_activity_score', 'narrative_similarity',
           'sector_divergence']
    for col in REQ:
        if col not in result.columns:
            result[col] = 0
    if 'timestamp' not in result.columns:
        result['timestamp'] = pd.Timestamp.now(tz='UTC')

    return result[REQ]


def collect_all(tickers: list[str], mode: str = 'both'):
    if not RAPIDAPI_KEY:
        logger.error(
            "RAPIDAPI_KEY not set in .env!\n"
            "  1. Go to https://rapidapi.com/ and create a free account\n"
            "  2. Subscribe to 'Reddit Search API' and 'Finance Social Sentiment'\n"
            "  3. Add RAPIDAPI_KEY=your_key to your .env file\n"
        )
        return

    reddit     = RedditRapidAPI(RAPIDAPI_KEY)
    stocktwits = StockTwitsRapidAPI(RAPIDAPI_KEY)

    logger.info(f"Collecting social data for {len(tickers)} tickers via RapidAPI...")
    logger.info(f"Mode: {mode}")
    logger.info("=" * 60)

    success, failed = [], []
    for ticker in tickers:
        try:
            features = collect_ticker(ticker, reddit, stocktwits, mode)
            if features is not None and not features.empty:
                output_path = SOCIAL_DIR / f'{ticker}_social.csv'
                features.to_csv(output_path, index=False)
                logger.info(f"[{ticker}] Saved {len(features)} rows -> {output_path}")
                success.append(ticker)
            else:
                logger.warning(f"[{ticker}] No data collected")
                failed.append(ticker)
        except Exception as e:
            logger.error(f"[{ticker}] Unexpected error: {e}")
            failed.append(ticker)

        time.sleep(2)   # Polite rate limiting between tickers

    logger.info("\n" + "=" * 60)
    logger.info(f"Done! Collected: {success}")
    if failed:
        logger.warning(f"Failed/empty: {failed}")
    logger.info(f"\nNext step: Run preprocessing to use real social data:")
    logger.info(f"  python scripts/03_preprocess_data.py")
    logger.info(f"  python scripts/04_train_model.py --epochs 80")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collect Reddit + StockTwits social data via RapidAPI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--tickers', nargs='+',
        default=['GME', 'AMC', 'TSLA', 'AAPL', 'MSFT', 'NVDA', 'BBBY', 'RIVN', 'MULN', 'PROG'],
        help='Tickers to collect data for'
    )
    parser.add_argument(
        '--mode', choices=['both', 'reddit-only', 'stocktwits-only'],
        default='both', help='Which sources to collect'
    )
    args = parser.parse_args()

    collect_all(args.tickers, args.mode)
