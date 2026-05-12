"""
praw_social_collector.py
========================
Collects Reddit social sentiment data using PRAW (official Reddit API)
and yfinance news as a backup source. This is FREE with no monthly limits.

SETUP (one-time):
  1. Go to https://www.reddit.com/prefs/apps
  2. Create a new "script" app:
     - Name: anything (e.g., "StockSentiment")
     - Redirect URI: http://localhost:8080
  3. Copy the client_id (under "personal use script") and client_secret
  4. Add to your .env file:
       REDDIT_CLIENT_ID=your_client_id
       REDDIT_CLIENT_SECRET=your_client_secret
       REDDIT_USER_AGENT=StockSentiment/1.0 by YourUsername

  Note: PRAW has very generous rate limits (60 req/min) and NO monthly cap.

OUTPUT:
  data/raw/social/{ticker}_social.csv  with columns:
    timestamp, ticker, sentiment_score, post_volume, engagement_score,
    bot_activity_score, narrative_similarity, sector_divergence

Usage:
    python scripts/praw_social_collector.py
    python scripts/praw_social_collector.py --tickers GME AMC TSLA
    python scripts/praw_social_collector.py --mode yfinance-only
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
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

SUBREDDITS = ['wallstreetbets', 'stocks', 'StockMarket', 'investing', 'pennystocks', 'options']
POSTS_PER_SUBREDDIT = 100
BULLISH_WORDS = ['moon', 'rocket', 'buy', 'calls', 'bullish', 'squeeze', 'pump',
                 'long', 'yolo', 'tendies', 'ath', 'breakout', 'upward', 'surge']
BEARISH_WORDS = ['crash', 'dump', 'puts', 'bearish', 'short', 'sell', 'overvalued',
                 'bubble', 'dead', 'rekt', 'bag', 'loss', 'warning', 'fraud', 'manipulation']


# ─────────────────────────────────────────────────────────────────────────────
# PRAW Reddit Collector
# ─────────────────────────────────────────────────────────────────────────────

def create_reddit_client():
    """Create a PRAW Reddit client from .env credentials."""
    try:
        import praw
        client_id     = os.getenv('REDDIT_CLIENT_ID', '')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET', '')
        user_agent    = os.getenv('REDDIT_USER_AGENT', 'StockSentiment/1.0')

        if not client_id or client_id == 'your_reddit_client_id':
            logger.warning("REDDIT_CLIENT_ID not configured. PRAW skipped.")
            return None

        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        # Quick test
        _ = reddit.subreddit('wallstreetbets').hot(limit=1)
        logger.info("PRAW Reddit client initialized successfully")
        return reddit
    except ImportError:
        logger.info("praw not installed. Installing...")
        os.system('pip install praw -q')
        return create_reddit_client()
    except Exception as e:
        logger.warning(f"PRAW init failed: {e}")
        return None


def collect_reddit_posts(reddit, ticker: str) -> pd.DataFrame:
    """Collect posts from multiple finance subreddits mentioning a ticker."""
    all_posts = []
    query = f'${ticker} OR "{ticker}" stock'

    for subreddit_name in SUBREDDITS:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            for submission in subreddit.search(query, sort='new', time_filter='month',
                                               limit=POSTS_PER_SUBREDDIT):
                all_posts.append({
                    'timestamp':    datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                    'title':        submission.title,
                    'score':        submission.score,
                    'num_comments': submission.num_comments,
                    'upvote_ratio': submission.upvote_ratio,
                    'subreddit':    subreddit_name,
                    'author':       str(submission.author) if submission.author else '[deleted]',
                })
            time.sleep(0.5)  # Respect rate limits
        except Exception as e:
            logger.warning(f"  Error fetching {subreddit_name} for ${ticker}: {e}")

    if not all_posts:
        return pd.DataFrame()

    df = pd.DataFrame(all_posts).drop_duplicates(subset=['title', 'timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"  PRAW: {len(df)} posts from {len(SUBREDDITS)} subreddits for ${ticker}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# yfinance News Collector (backup + supplement)
# ─────────────────────────────────────────────────────────────────────────────

def collect_yfinance_news(ticker: str) -> pd.DataFrame:
    """Collect recent news headlines from yfinance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        news = stock.news or []
        if not news:
            return pd.DataFrame()

        rows = []
        for item in news:
            content = item.get('content', {})
            title   = content.get('title', item.get('title', ''))
            pubdate = content.get('pubDate', '')
            prov_ts = item.get('providerPublishTime', 0)

            if pubdate:
                try:
                    ts = pd.to_datetime(pubdate, utc=True)
                except Exception:
                    ts = pd.Timestamp.now(tz='UTC')
            elif prov_ts:
                ts = pd.to_datetime(prov_ts, unit='s', utc=True)
            else:
                ts = pd.Timestamp.now(tz='UTC')

            rows.append({
                'timestamp':    ts,
                'title':        title,
                'score':        10,   # News gets default engagement
                'num_comments': 0,
                'upvote_ratio': 0.8,
                'subreddit':    'news',
                'author':       item.get('publisher', 'unknown'),
            })

        df = pd.DataFrame(rows).drop_duplicates(subset=['title'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"  yfinance: {len(df)} news articles for ${ticker}")
        return df
    except Exception as e:
        logger.warning(f"  yfinance news error for ${ticker}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering: raw posts → training feature format
# ─────────────────────────────────────────────────────────────────────────────

def posts_to_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert raw Reddit/news posts DataFrame to hourly social features.

    Key improvement over naive zero-fill:
    - Forward-fill sentiment for 48h (news effect persists)
    - Exponential decay applied after forward-fill
    - Add 24h rolling sentiment + volume features
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp'])

    title = df['title'].fillna('').str.lower()
    df['bullish'] = title.apply(lambda t: sum(1 for w in BULLISH_WORDS if w in t))
    df['bearish'] = title.apply(lambda t: sum(1 for w in BEARISH_WORDS if w in t))
    df['raw_sent'] = (df['bullish'] - df['bearish']).clip(-3, 3) / 3.0

    # Resample to hourly bins — posts per hour
    df = df.set_index('timestamp').sort_index()
    hourly = df.resample('1h').agg({
        'raw_sent':      'mean',
        'score':         'sum',
        'num_comments':  'sum',
        'upvote_ratio':  'mean',
        'bullish':       'sum',
        'bearish':       'sum',
    })  # NaN = no posts that hour

    post_count = df.resample('1h').size().rename('post_volume')
    hourly = hourly.join(post_count)

    # ── KEY FIX: forward-fill sentiment with exponential decay ──────────────
    # Instead of zero-fill (sparse noise), keep the last known sentiment signal
    # decaying it by 0.88/hour (half-life ≈ 12h). For yfinance's ~10 articles
    # this turns ~1% populated hours into 100% dense signal.
    hourly['raw_sent'] = hourly['raw_sent'].ffill(limit=48).fillna(0)
    sent_series = hourly['raw_sent'].copy()
    was_filled = hourly['post_volume'].isna() | (hourly['post_volume'] == 0)
    decay = 1.0
    prev_val = 0.0
    sent_filled = []
    for val, filled in zip(sent_series, was_filled):
        if not filled and not np.isnan(val):
            prev_val = val
            decay = 1.0
            sent_filled.append(val)
        else:
            decay *= 0.88  # ~12h half-life (0.88^12 ≈ 0.22)
            sent_filled.append(prev_val * decay)
    hourly['sentiment_score'] = np.clip(sent_filled, -1, 1)

    # Fill remaining NaN cols with 0
    hourly = hourly.fillna(0)

    # ── Derived features ─────────────────────────────────────────────────────
    hourly['engagement_score']    = (hourly['score'] + hourly['num_comments'] * 2).clip(0)
    hourly['bot_activity_score']  = (1 - hourly['upvote_ratio'].where(hourly['upvote_ratio'] > 0, 0.8)).clip(0, 1)
    total_mentions = hourly['bullish'] + hourly['bearish'] + 1e-9
    hourly['narrative_similarity'] = (
        (hourly['bullish'].abs() / total_mentions).clip(0, 1) * 0.7
        + (1 - hourly['bot_activity_score']) * 0.3
    )
    hourly['sector_divergence'] = 0.0

    # ── Rolling features (24h window) ────────────────────────────────────────
    # These give the model a "recent trend" dimension beyond point-in-time
    hourly['rolling_sentiment_24h'] = hourly['sentiment_score'].rolling(24, min_periods=1).mean()
    hourly['rolling_volume_24h']    = hourly['post_volume'].rolling(24, min_periods=1).mean()

    hourly['ticker'] = ticker
    result = hourly[['ticker', 'sentiment_score', 'post_volume', 'engagement_score',
                      'bot_activity_score', 'narrative_similarity', 'sector_divergence',
                      'rolling_sentiment_24h', 'rolling_volume_24h']].reset_index()
    result = result.rename(columns={'timestamp': 'timestamp'})
    result['timestamp'] = pd.to_datetime(result['timestamp'], utc=True)

    logger.info(f"  Features: {len(result)} hourly rows for ${ticker} "
                f"(non-zero sentiment: {(result['sentiment_score'] != 0).sum()})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main collection pipeline
# ─────────────────────────────────────────────────────────────────────────────

def collect_ticker(ticker: str, reddit=None, mode: str = 'both') -> pd.DataFrame:
    """Collect and merge Reddit + yfinance news features for one ticker."""
    all_posts = []

    if mode in ('both', 'reddit-only') and reddit is not None:
        logger.info(f"  Fetching PRAW Reddit posts...")
        reddit_df = collect_reddit_posts(reddit, ticker)
        if not reddit_df.empty:
            all_posts.append(reddit_df)

    if mode in ('both', 'yfinance-only'):
        logger.info(f"  Fetching yfinance news...")
        news_df = collect_yfinance_news(ticker)
        if not news_df.empty:
            all_posts.append(news_df)

    if not all_posts:
        logger.warning(f"[{ticker}] No posts from any source")
        return pd.DataFrame()

    combined = pd.concat(all_posts, ignore_index=True)
    combined = combined.drop_duplicates(subset=['title'])
    logger.info(f"  Combined: {len(combined)} total posts for ${ticker}")
    return posts_to_features(combined, ticker)


def collect_all(tickers: list, mode: str = 'both'):
    """Run collection for all tickers."""
    reddit = None
    if mode in ('both', 'reddit-only'):
        reddit = create_reddit_client()
        if reddit is None and mode == 'reddit-only':
            logger.warning("PRAW unavailable. Switching to yfinance-only mode.")
            mode = 'yfinance-only'

    logger.info(f"Collecting social data for {len(tickers)} tickers...")
    logger.info(f"Mode: {mode}")
    logger.info("=" * 60)

    success, failed = [], []
    for ticker in tickers:
        logger.info(f"\n[{ticker}]")
        try:
            features = collect_ticker(ticker, reddit, mode)
            if features is not None and not features.empty:
                output_path = SOCIAL_DIR / f'{ticker}_social.csv'
                features.to_csv(output_path, index=False)
                logger.info(f"  Saved {len(features)} rows -> {output_path}")
                success.append(ticker)
            else:
                logger.warning(f"  No data collected")
                failed.append(ticker)
        except Exception as e:
            logger.error(f"  Error: {e}")
            failed.append(ticker)
        time.sleep(1)

    logger.info("\n" + "=" * 60)
    logger.info(f"Done! Collected: {success}")
    if failed:
        logger.warning(f"Failed/empty: {failed}")
    logger.info(f"\nNext steps:")
    logger.info(f"  python scripts/03_preprocess_data.py")
    logger.info(f"  python scripts/04_train_model.py --epochs 80")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collect Reddit + news social data via PRAW + yfinance',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--tickers', nargs='+',
        default=['GME', 'AMC', 'TSLA', 'AAPL', 'MSFT', 'NVDA', 'BBBY', 'RIVN', 'MULN', 'PROG'],
        help='Tickers to collect data for'
    )
    parser.add_argument(
        '--mode', choices=['both', 'reddit-only', 'yfinance-only'],
        default='both', help='Which sources to collect'
    )
    args = parser.parse_args()
    collect_all(args.tickers, args.mode)
