"""
Social Sentiment Collector — uses Finnhub's social sentiment endpoint which
aggregates Reddit + Twitter mentions automatically. No extra API key needed,
just the FINNHUB_API_KEY already in .env.

Finnhub /stock/social-sentiment endpoint:
  - Returns Reddit mention counts, positive/negative mention counts
  - Covers the last 7 days
  - Free tier: 60 req/min

Fallback chain:
  1. Finnhub social sentiment (Reddit + Twitter aggregated)
  2. yfinance news headlines (keyword sentiment)
  3. Neutral defaults [0.5, 0, 0, 0, 0, 0]

6-feature vector (matches model social input):
    [sentiment_score, post_volume, engagement_score,
     bot_activity_score, narrative_similarity, pump_signal]
"""

import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"

_PUMP_WORDS = {
    'moon', 'rocket', 'squeeze', 'yolo', 'lambo', 'ape', 'diamond hands',
    'hold', 'short squeeze', 'gamma squeeze', 'tendies', '🚀', '💎',
    'to the moon', '100x', '10x', 'parabolic', 'breaking out', 'catalyst',
    'buy the dip', 'calls only', 'buying more', 'short squeeze',
}
_BULLISH_WORDS = {
    'bullish', 'calls', 'long', 'buy', 'accumulate', 'support',
    'breakout', 'undervalued', 'upside', 'strong', 'momentum', 'surge',
}
_BEARISH_WORDS = {
    'puts', 'short', 'bearish', 'sell', 'dump', 'overvalued',
    'avoid', 'fraud', 'scam', 'declining', 'weak', 'fall',
}

_last_request: float = 0.0
_MIN_INTERVAL = 1.2   # seconds between Finnhub calls


def _throttle():
    global _last_request
    elapsed = time.time() - _last_request
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request = time.time()


def _headline_sentiment(text: str) -> dict:
    t = text.lower()
    pump  = min(sum(1 for w in _PUMP_WORDS   if w in t) / 2.0, 1.0)
    bull  = any(w in t for w in _BULLISH_WORDS)
    bear  = any(w in t for w in _BEARISH_WORDS)
    return {'pump': pump, 'bullish': bull, 'bearish': bear}


class StockTwitsCollector:
    """
    Finnhub-based social sentiment collector.
    Named StockTwitsCollector to maintain interface compatibility.
    Uses Finnhub's aggregated Reddit/Twitter sentiment endpoint.
    """

    def __init__(self):
        self.api_key = os.getenv('FINNHUB_API_KEY', '')
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({'X-Finnhub-Token': self.api_key})
            logger.info("Finnhub social sentiment collector initialised")
        else:
            logger.warning("FINNHUB_API_KEY not set — social features will use yfinance news fallback")

        self._cache: dict = {}
        self._cache_ttl = 300  # 5-minute cache

    def _get_social_sentiment(self, ticker: str) -> Optional[dict]:
        """Fetch Finnhub social sentiment (Reddit + Twitter aggregated)."""
        if not self.api_key:
            return None
        _throttle()
        try:
            # Finnhub social sentiment: last 7 days of Reddit & Twitter data
            today  = datetime.now().strftime('%Y-%m-%d')
            week   = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            resp   = self.session.get(
                f"{FINNHUB_BASE}/stock/social-sentiment",
                params={'symbol': ticker.split('.')[0], 'from': week, 'to': today},
                timeout=8,
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"Finnhub social sentiment {resp.status_code} for {ticker}")
        except Exception as e:
            logger.warning(f"Finnhub social sentiment error: {e}")
        return None

    def _get_news_sentiment(self, ticker: str) -> list:
        """Fallback: fetch recent news headlines via yfinance."""
        try:
            import yfinance as yf
            news = yf.Ticker(ticker).news or []
            return [item.get('content', {}).get('title', '') or item.get('title', '')
                    for item in news[:20]]
        except Exception:
            return []

    def build_social_features(self, ticker: str) -> np.ndarray:
        """
        6-element feature vector:
            [sentiment_score, post_volume, engagement_score,
             bot_activity_score, narrative_similarity, pump_signal]
        """
        now = time.time()
        if ticker in self._cache:
            ts, feats = self._cache[ticker]
            if now - ts < self._cache_ttl:
                return feats

        # ── Try Finnhub social sentiment first ────────────────────────────────
        social = self._get_social_sentiment(ticker)
        if social:
            reddit_data  = social.get('reddit',  [])
            twitter_data = social.get('twitter', [])
            all_data = reddit_data + twitter_data

            if all_data:
                total_mentions  = sum(d.get('mention', 0)         for d in all_data)
                positive        = sum(d.get('positiveMention', 0) for d in all_data)
                negative        = sum(d.get('negativeMention', 0) for d in all_data)
                labelled        = positive + negative

                sentiment_score = (positive / labelled) if labelled > 0 else 0.5

                # Volume: normalise — 500 mentions/week = 1.0
                post_volume = min(total_mentions / 500.0, 1.0)

                # Engagement: use mention score from Finnhub (0-1 normalised)
                engagement_score = min(
                    sum(d.get('score', 0) for d in all_data) / (len(all_data) * 10 + 1e-9),
                    1.0
                )

                # Bot activity: rapid mention spikes (score drops but volume spikes)
                if len(all_data) >= 2:
                    mentions_list = [d.get('mention', 0) for d in all_data[-7:]]
                    if max(mentions_list) > 0:
                        spike = max(mentions_list) / (np.mean(mentions_list) + 1e-9)
                        bot_activity_score = min((spike - 1) / 4.0, 1.0)
                    else:
                        bot_activity_score = 0.0
                else:
                    bot_activity_score = 0.0

                # Narrative similarity: derived from sentiment consistency
                if labelled > 0:
                    consistency = abs(positive - negative) / labelled
                    narrative_similarity = consistency * 0.5  # scale down
                else:
                    narrative_similarity = 0.0

                # Pump signal: high positive sentiment + high volume = pump-like
                pump_signal = sentiment_score * post_volume

                feats = np.array([
                    sentiment_score, post_volume, engagement_score,
                    bot_activity_score, narrative_similarity, pump_signal,
                ], dtype=np.float32)

                logger.info(
                    f"Finnhub social for {ticker}: "
                    f"sent={sentiment_score:.2f} vol={post_volume:.2f} "
                    f"eng={engagement_score:.2f} bot={bot_activity_score:.2f} "
                    f"pump={pump_signal:.2f} "
                    f"(reddit={len(reddit_data)} twitter={len(twitter_data)} entries)"
                )
                self._cache[ticker] = (now, feats)
                return feats

        # ── Fallback: yfinance news headlines ────────────────────────────────
        headlines = self._get_news_sentiment(ticker)
        if headlines:
            scored = [_headline_sentiment(h) for h in headlines]
            bullish = sum(1 for s in scored if s['bullish'] and not s['bearish'])
            bearish = sum(1 for s in scored if s['bearish'] and not s['bullish'])
            labelled = bullish + bearish
            sentiment_score = (bullish / labelled) if labelled > 0 else 0.5
            pump_signal = float(np.mean([s['pump'] for s in scored]))
            post_volume = min(len(headlines) / 20.0, 1.0)

            feats = np.array([
                sentiment_score, post_volume, 0.2,
                0.0, 0.0, pump_signal,
            ], dtype=np.float32)

            logger.info(
                f"yfinance news fallback for {ticker}: "
                f"sent={sentiment_score:.2f} vol={post_volume:.2f} pump={pump_signal:.2f}"
            )
            self._cache[ticker] = (now, feats)
            return feats

        # ── Final fallback: neutral defaults ──────────────────────────────────
        logger.warning(f"No social data available for {ticker} — using neutral defaults")
        feats = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._cache[ticker] = (now, feats)
        return feats
