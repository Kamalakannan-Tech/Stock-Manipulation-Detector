"""
rss_social_collector.py
=======================
Collects stock sentiment data from FREE sources — NO API KEY REQUIRED.

Sources used:
  1. Google News RSS    — 50-100 articles per ticker, last 30 days
  2. Yahoo Finance RSS  — 10-20 articles per ticker
  3. Finviz news        — 20-40 articles per ticker
  4. yfinance news      — 10 articles per ticker (backup)

OUTPUT:
  data/raw/social/{ticker}_social.csv  compatible with existing pipeline

Usage:
    python scripts/rss_social_collector.py
    python scripts/rss_social_collector.py --tickers GME AMC TSLA
"""

import os, sys, time, logging, argparse, re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.parse import quote
import xml.etree.ElementTree as ET

sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SOCIAL_DIR = Path('data/raw/social')
SOCIAL_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

BULLISH_WORDS = [
    'surge', 'rally', 'gain', 'rise', 'jump', 'soar', 'moon', 'rocket', 'buy',
    'bullish', 'squeeze', 'pump', 'breakout', 'upward', 'record', 'beat', 'profit',
    'growth', 'strong', 'upgrade', 'outperform', 'acquisition', 'deal', 'positive'
]
BEARISH_WORDS = [
    'crash', 'drop', 'fall', 'plunge', 'dump', 'bearish', 'short', 'sell',
    'fraud', 'manipulation', 'loss', 'miss', 'decline', 'warning', 'downgrade',
    'lawsuit', 'sec', 'investigation', 'bankrupt', 'layoff', 'recall', 'negative'
]

MANIPULATION_WORDS = [
    'pump', 'dump', 'squeeze', 'manipulation', 'sec', 'investigation',
    'fraud', 'scheme', 'unusual volume', 'halt', 'spike'
]


# ─── Fetchers ────────────────────────────────────────────────────────────────

def fetch_url(url: str, timeout: int = 10) -> str:
    """Fetch a URL with a browser-like User-Agent."""
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='replace')
    except Exception as e:
        logger.debug(f"  fetch_url error {url[:60]}: {e}")
        return ''


def parse_rss_date(date_str: str) -> pd.Timestamp:
    """Parse an RSS pubDate string to a UTC Timestamp."""
    if not date_str:
        return pd.Timestamp.now(tz='UTC')
    for fmt in (
        '%a, %d %b %Y %H:%M:%S %z',
        '%a, %d %b %Y %H:%M:%S GMT',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%dT%H:%M:%SZ',
    ):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return pd.Timestamp(dt).tz_convert('UTC')
        except ValueError:
            continue
    return pd.Timestamp.now(tz='UTC')


def fetch_google_news_rss(ticker: str) -> list[dict]:
    """Fetch from Google News RSS — no key, returns ~50-100 items."""
    query = quote(f'"{ticker}" stock OR shares OR market')
    url   = f'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en'
    xml   = fetch_url(url)
    if not xml:
        return []

    try:
        root  = ET.fromstring(xml)
        items = root.findall('./channel/item')
        rows  = []
        for item in items:
            title   = item.findtext('title', '').strip()
            pubdate = item.findtext('pubDate', '')
            ts      = parse_rss_date(pubdate)
            rows.append({'timestamp': ts, 'title': title, 'source': 'google_news'})
        logger.info(f"  Google News RSS: {len(rows)} articles for ${ticker}")
        return rows
    except ET.ParseError:
        return []


def fetch_yahoo_finance_rss(ticker: str) -> list[dict]:
    """Fetch from Yahoo Finance RSS feed — no key needed."""
    url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
    xml = fetch_url(url)
    if not xml:
        return []

    try:
        root  = ET.fromstring(xml)
        items = root.findall('./channel/item')
        rows  = []
        for item in items:
            title   = item.findtext('title', '').strip()
            pubdate = item.findtext('pubDate', '')
            ts      = parse_rss_date(pubdate)
            rows.append({'timestamp': ts, 'title': title, 'source': 'yahoo_rss'})
        logger.info(f"  Yahoo Finance RSS: {len(rows)} articles for ${ticker}")
        return rows
    except ET.ParseError:
        return []


def fetch_finviz_news(ticker: str) -> list[dict]:
    """Scrape Finviz news table — publicly available, no login needed."""
    url  = f'https://finviz.com/quote.ashx?t={ticker}&p=d'
    html = fetch_url(url)
    if not html:
        return []

    # Finviz stores news as <td class="news_date"> / <td class="news_link">
    pattern = re.compile(
        r'<tr[^>]*>\s*<td[^>]*class="news_date[^"]*"[^>]*>(.*?)</td>'
        r'.*?<td[^>]*class="news_link[^"]*"[^>]*>.*?<a[^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE
    )
    rows = []
    for m in pattern.finditer(html):
        raw_date = re.sub('<[^>]+>', '', m.group(1)).strip()
        title    = re.sub('<[^>]+>', '', m.group(2)).strip()
        if not title:
            continue
        try:
            # Finviz dates: "Mar-24-24 09:30AM" or "Mar-24-24"
            if 'AM' in raw_date or 'PM' in raw_date:
                ts = pd.to_datetime(raw_date, format='%b-%d-%y %I:%M%p', utc=True)
            else:
                ts = pd.to_datetime(raw_date, format='%b-%d-%y', utc=True)
        except Exception:
            ts = pd.Timestamp.now(tz='UTC')
        rows.append({'timestamp': ts, 'title': title, 'source': 'finviz'})

    logger.info(f"  Finviz: {len(rows)} articles for ${ticker}")
    return rows


def fetch_yfinance_news(ticker: str) -> list[dict]:
    """yfinance news as fallback."""
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news or []
        rows = []
        for item in news:
            content = item.get('content', {})
            title   = content.get('title', item.get('title', ''))
            pubdate = content.get('pubDate', '')
            prov_ts = item.get('providerPublishTime', 0)
            if pubdate:
                try:    ts = pd.to_datetime(pubdate, utc=True)
                except: ts = pd.Timestamp.now(tz='UTC')
            elif prov_ts:
                ts = pd.to_datetime(prov_ts, unit='s', utc=True)
            else:
                ts = pd.Timestamp.now(tz='UTC')
            rows.append({'timestamp': ts, 'title': title, 'source': 'yfinance'})
        logger.info(f"  yfinance: {len(rows)} articles for ${ticker}")
        return rows
    except Exception as e:
        logger.debug(f"  yfinance error: {e}")
        return []


# ─── Feature engineering ─────────────────────────────────────────────────────

def articles_to_features(articles: list[dict], ticker: str) -> pd.DataFrame:
    """Convert list of article dicts to hourly social feature rows."""
    if not articles:
        return pd.DataFrame()

    df = pd.DataFrame(articles)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp'])
    if df.empty:
        return pd.DataFrame()

    title = df['title'].fillna('').str.lower()
    df['bullish']       = title.apply(lambda t: sum(1 for w in BULLISH_WORDS      if w in t))
    df['bearish']       = title.apply(lambda t: sum(1 for w in BEARISH_WORDS      if w in t))
    df['manipulation']  = title.apply(lambda t: sum(1 for w in MANIPULATION_WORDS if w in t))
    df['raw_sent']      = (df['bullish'] - df['bearish']).clip(-3, 3) / 3.0

    # Aggregate to hourly
    df = df.set_index('timestamp').sort_index()
    hourly = df.resample('1h').agg(
        raw_sent=('raw_sent', 'mean'),
        score=('bullish', 'sum'),       # repurpose 'score' as bullish count
        num_comments=('manipulation', 'sum'),   # repurpose as manipulation signal
        bullish=('bullish', 'sum'),
        bearish=('bearish', 'sum'),
    )
    post_count = df.resample('1h').size().rename('post_volume')
    hourly = hourly.join(post_count)

    # Forward-fill with exponential decay (0.88/hr ≈ 12h half-life)
    sent_series = hourly['raw_sent'].copy()
    was_filled  = hourly['post_volume'].isna() | (hourly['post_volume'] == 0)
    decay, prev = 1.0, 0.0
    sent_filled = []
    for val, filled in zip(sent_series, was_filled):
        if not filled and not (isinstance(val, float) and np.isnan(val)):
            prev, decay = val, 1.0
            sent_filled.append(val)
        else:
            decay *= 0.88
            sent_filled.append(prev * decay)

    hourly = hourly.fillna(0)
    hourly['sentiment_score']     = np.clip(sent_filled, -1, 1)
    hourly['engagement_score']    = hourly['score'].clip(0)
    hourly['bot_activity_score']  = 0.1   # news articles are low-bot
    total = hourly['bullish'] + hourly['bearish'] + 1e-9
    hourly['narrative_similarity'] = (hourly['bullish'] / total).clip(0, 1)
    hourly['sector_divergence']   = 0.0
    hourly['rolling_sentiment_24h'] = hourly['sentiment_score'].rolling(24, min_periods=1).mean()
    hourly['rolling_volume_24h']  = hourly['post_volume'].rolling(24, min_periods=1).mean()
    hourly['ticker'] = ticker

    result = hourly[['ticker', 'sentiment_score', 'post_volume', 'engagement_score',
                      'bot_activity_score', 'narrative_similarity', 'sector_divergence',
                      'rolling_sentiment_24h', 'rolling_volume_24h']].reset_index()
    result['timestamp'] = pd.to_datetime(result['timestamp'], utc=True)

    non_zero = (result['sentiment_score'] != 0).sum()
    logger.info(f"  → {len(result)} hourly rows (non-zero sentiment: {non_zero}/{len(result)})")
    return result


# ─── Main pipeline ───────────────────────────────────────────────────────────

def collect_ticker(ticker: str) -> pd.DataFrame:
    all_articles = []

    # 1. Google News RSS (best source — most articles)
    all_articles.extend(fetch_google_news_rss(ticker))
    time.sleep(1)

    # 2. Yahoo Finance RSS
    all_articles.extend(fetch_yahoo_finance_rss(ticker))
    time.sleep(0.5)

    # 3. Finviz news
    all_articles.extend(fetch_finviz_news(ticker))
    time.sleep(1)

    # 4. yfinance as fallback
    all_articles.extend(fetch_yfinance_news(ticker))

    # Deduplicate by title
    seen, unique = set(), []
    for a in all_articles:
        key = a['title'].lower()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(a)

    logger.info(f"  [{ticker}] Total unique articles: {len(unique)}")
    return articles_to_features(unique, ticker)


def collect_all(tickers: list):
    logger.info(f"Collecting news sentiment for {len(tickers)} tickers (no API key needed)...")
    logger.info("=" * 60)

    success, failed = [], []
    for ticker in tickers:
        logger.info(f"\n── {ticker} ──────────────────────────────")
        try:
            features = collect_ticker(ticker)
            if features is not None and not features.empty:
                out = SOCIAL_DIR / f'{ticker}_social.csv'
                features.to_csv(out, index=False)
                logger.info(f"  ✓ Saved → {out}")
                success.append(ticker)
            else:
                logger.warning(f"  ✗ No data")
                failed.append(ticker)
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            failed.append(ticker)
        time.sleep(1.5)

    logger.info("\n" + "=" * 60)
    logger.info(f"Done! ✓ {success}")
    if failed:
        logger.warning(f"Failed: {failed}")
    logger.info("\nNext:")
    logger.info("  python scripts/03_preprocess_data.py")
    logger.info("  python scripts/04_train_model.py --epochs 100")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect stock news from free RSS sources (no API key)')
    parser.add_argument('--tickers', nargs='+',
        default=['GME','AMC','TSLA','AAPL','MSFT','NVDA','BBBY','RIVN','MULN','PROG'])
    args = parser.parse_args()
    collect_all(args.tickers)
