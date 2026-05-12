"""
Finnhub data collector — real-time news sentiment and stock quotes for US stocks.

Free tier:
  - 60 API calls/minute
  - Real-time US stock quotes (GLOBAL_QUOTE equivalent)
  - Company news (last 30 days)
  - Sentiment scores for social media buzz

Finnhub does NOT cover Indian stocks (NSE/BSE) on the free tier.
For Indian stocks, yfinance is used automatically.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional
import requests

logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"

# Simple rate throttle: free tier = 60 req/min → 1 req/sec
_last_request_time: float = 0.0
_MIN_INTERVAL = 1.1  # seconds between requests


def _throttle():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


class FinnhubCollector:
    """
    Fetches news sentiment and real-time quotes from Finnhub.
    Used to provide REAL social/sentiment features to replace market-derived proxies.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY", "")
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY not set")
        self.session = requests.Session()
        self.session.headers.update({"X-Finnhub-Token": self.api_key})

    def _get(self, endpoint: str, params: dict) -> Optional[dict]:
        """Make a throttled GET request to Finnhub."""
        _throttle()
        try:
            resp = self.session.get(f"{FINNHUB_BASE}/{endpoint}", params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            logger.error(f"Finnhub HTTP error ({endpoint}): {e}")
        except requests.RequestException as e:
            logger.error(f"Finnhub request error ({endpoint}): {e}")
        except Exception as e:
            logger.error(f"Finnhub unexpected error ({endpoint}): {e}")
        return None

    def get_news_sentiment(self, ticker: str) -> Optional[dict]:
        """
        Fetch social media sentiment buzz score for a ticker.
        Returns dict with keys: buzz, sentiment, articlesInLastWeek, weeklyAverage.

        buzz score:   > 1.0 = above-average social activity (potential pump signal)
        sentiment:    0-1   = positive/negative ratio
        """
        # Strip exchange suffix for US tickers (Finnhub only handles plain symbols)
        symbol = ticker.split(".")[0] if "." in ticker else ticker
        data = self._get("news-sentiment", {"symbol": symbol})
        if not data or "buzz" not in data:
            return None
        buzz   = data.get("buzz", {})
        sent   = data.get("sentiment", {})
        return {
            "ticker":                ticker,
            "buzz_score":            float(buzz.get("buzz", 0)),
            "articles_in_last_week": int(buzz.get("articlesInLastWeek", 0)),
            "weekly_average":        float(buzz.get("weeklyAverage", 0)),
            "company_news_score":    float(buzz.get("companyNewsScore", 0)),
            "bearish_pct":           float(sent.get("bearishPercent", 0)),
            "bullish_pct":           float(sent.get("bullishPercent", 0)),
            "sector_avg_bearish":    float(data.get("sectorAverageBullishPercent", 0)),
            "timestamp":             datetime.now().isoformat(),
        }

    def get_company_news(self, ticker: str, days_back: int = 3) -> list:
        """
        Fetch recent company news headlines + sentiment for a ticker.
        Returns list of news items sorted by datetime (newest first).
        """
        symbol = ticker.split(".")[0] if "." in ticker else ticker
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date   = datetime.now().strftime("%Y-%m-%d")
        data = self._get("company-news", {
            "symbol": symbol,
            "from":   from_date,
            "to":     to_date,
        })
        if not data or not isinstance(data, list):
            return []
        # Return relevant fields only
        return [
            {
                "headline":  item.get("headline", ""),
                "source":    item.get("source", ""),
                "datetime":  item.get("datetime", 0),
                "sentiment": item.get("sentiment", "neutral"),
                "url":       item.get("url", ""),
            }
            for item in data[:20]  # cap at 20 items
        ]

    def get_quote(self, ticker: str) -> Optional[dict]:
        """
        Fetch the current real-time quote for a ticker.
        Returns dict with: price, open, high, low, prev_close, change_pct, timestamp.
        """
        symbol = ticker.split(".")[0] if "." in ticker else ticker
        data = self._get("quote", {"symbol": symbol})
        if not data or "c" not in data:
            return None
        c = data.get("c", 0)
        pc = data.get("pc", 1)
        return {
            "ticker":      ticker,
            "price":       float(c),
            "open":        float(data.get("o", 0)),
            "high":        float(data.get("h", 0)),
            "low":         float(data.get("l", 0)),
            "prev_close":  float(pc),
            "change_pct":  round((c - pc) / (pc + 1e-9) * 100, 4),
            "timestamp":   datetime.fromtimestamp(data.get("t", 0)).isoformat(),
        }

    # Bullish and bearish keyword lists for headline scoring
    _BULLISH = {'surge', 'soar', 'rally', 'beat', 'record', 'gain', 'rise', 'bull',
                'buy', 'upgrade', 'positive', 'growth', 'profit', 'strong', 'high',
                'breakout', 'opportunity', 'outperform', 'momentum', 'boost'}
    _BEARISH = {'drop', 'fall', 'plunge', 'crash', 'loss', 'miss', 'weak', 'bear',
                'sell', 'downgrade', 'negative', 'decline', 'warning', 'risk',
                'low', 'short', 'concern', 'lawsuit', 'fraud', 'manipulation',
                'investigation', 'lawsuit', 'probe', 'cut', 'disappoint'}

    def _score_headline(self, headline: str) -> float:
        """Return simple sentiment score: positive=+1, negative=-1, neutral=0."""
        words = set(headline.lower().split())
        pos = len(words & self._BULLISH)
        neg = len(words & self._BEARISH)
        if pos == neg:
            return 0.0
        return 1.0 if pos > neg else -1.0

    def build_social_features(self, ticker: str) -> list:
        """
        Build a 6-element real social feature vector from free Finnhub endpoints.

        Uses company-news (free) for article count + keyword sentiment scoring.
        Feature order matches SocialFeatureExtractor output:
          [0] activity_norm   — article count vs 3/day baseline (buzz proxy)
          [1] net_sentiment   — (bullish_articles - bearish) / total
          [2] news_score      — recency-weighted avg sentiment (-1 to 1)
          [3] above_avg       — 1.0 if more articles than baseline, else 0.0
          [4] bearish_ratio   — fraction of bearish articles
          [5] vol_ratio       — articles in last 1 day vs last 3 days

        Returns zeros (neutral) if no articles are found.
        """
        news_3d = self.get_company_news(ticker, days_back=3)
        if not news_3d:
            logger.warning(f"No Finnhub company news for {ticker}, using neutral defaults")
            return [0.0] * 6

        now_ts = datetime.now().timestamp()
        one_day_ago = now_ts - 86400

        scores   = [self._score_headline(a['headline']) for a in news_3d]
        recent   = [a for a in news_3d if a['datetime'] >= one_day_ago]
        n_total  = len(scores)
        n_recent = len(recent)
        BASELINE = 3.0  # expected articles per day for normal stock

        bullish_n = scores.count(1.0)
        bearish_n = scores.count(-1.0)

        # [0] activity_norm: above 1.0 = unusually high coverage
        activity_norm  = min(n_recent / BASELINE, 3.0)
        # [1] net_sentiment: fraction difference (-1 to 1)
        net_sentiment  = (bullish_n - bearish_n) / (n_total + 1e-9)
        # [2] news_score: simple average sentiment
        news_score     = sum(scores) / (n_total + 1e-9)
        # [3] above_avg: 1 if recent > baseline
        above_avg      = 1.0 if n_recent > BASELINE else 0.0
        # [4] bearish_ratio
        bearish_ratio  = bearish_n / (n_total + 1e-9)
        # [5] vol_ratio: recent vs all (covers 3 days)
        vol_ratio      = min(n_recent / (n_total + 1e-9) * 3, 3.0)

        features = [
            float(activity_norm),
            float(net_sentiment),
            float(news_score),
            float(above_avg),
            float(bearish_ratio),
            float(vol_ratio),
        ]
        logger.info(
            f"Finnhub news [{ticker}]: {n_recent}/{n_total} articles today, "
            f"net_sent={net_sentiment:.2f}, bearish={bearish_ratio:.2f}"
        )
        return features

    def is_supported(self, ticker: str) -> bool:
        """Finnhub free tier supports US stocks only (no .NS/.BSE)."""
        return "." not in ticker
