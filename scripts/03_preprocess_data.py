"""
03_preprocess_data.py — Combines market data + social features, runs labeling,
and saves combined_features.csv + labels.csv.

Improvements:
  - Tries to load REAL Finnhub/Reddit social data first (from data/raw/social/)
  - Falls back to market-proxy social features if real data not available
  - Logs data source clearly so you know which mode is running
  - Confidence score distribution logged for label quality awareness
"""
import sys, os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

from src.preprocessing.feature_extractor import MarketFeatureExtractor, SocialFeatureExtractor
from src.preprocessing.labeler import ManipulationLabeler
from src.inference.sector_decoupler import SectorDecoupler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TICKERS      = ['GME','AMC','TSLA','AAPL','MSFT','NVDA','BBBY','RIVN','MULN','PROG']
RAW_DIR      = Path('data/raw/market')
SOCIAL_DIR   = Path('data/raw/social')
PROCESSED_DIR = Path('data/processed')
LABELED_DIR   = Path('data/labeled')

market_extractor = MarketFeatureExtractor()
social_extractor  = SocialFeatureExtractor()
labeler           = ManipulationLabeler(
    price_spike_threshold=0.10,
    volume_threshold=2.0,
    crash_threshold=-0.08,
    time_window_hours=48,
)
decoupler = SectorDecoupler()


def load_ticker(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f'{ticker}_market.csv'
    if not path.exists():
        logger.warning(f'No raw data for {ticker}')
        return pd.DataFrame()
    df = pd.read_csv(path)
    ts_col = ('timestamp' if 'timestamp' in df.columns else
              'Datetime'  if 'Datetime'  in df.columns else
              'Date'      if 'Date'      in df.columns else df.columns[0])
    df = df.rename(columns={ts_col: 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df['ticker'] = ticker
    return df


def load_real_social(ticker: str, n_rows: int) -> pd.DataFrame | None:
    """
    Attempt to load real social sentiment data from data/raw/social/.
    Returns a DataFrame aligned to n_rows if found, else None.
    
    Supported formats:
      - {ticker}_social.csv    : StockTwits / Reddit collector output
      - {ticker}_finnhub.csv   : Finnhub news sentiment collector output
    """
    SOCIAL_COLS_REQUIRED = ['sentiment_score', 'post_volume']

    candidates = [
        SOCIAL_DIR / f'{ticker}_social.csv',
        SOCIAL_DIR / f'{ticker}_finnhub.csv',
        SOCIAL_DIR / f'{ticker.lower()}_social.csv',
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            # Check if it has the minimum required real social columns
            if not all(c in df.columns for c in SOCIAL_COLS_REQUIRED):
                logger.debug(f"  {path.name}: missing required columns, skipping")
                continue
            # Check it's not just synthetic (synthetic files have bot_activity_score derived from volume z-score)
            # We detect synthetic by checking if sentiment_score is perfectly correlated with RSI
            if 'sentiment_score' in df.columns and len(df) > 20:
                # Simple heuristic: real data has more variance in post_volume
                pv = df['post_volume'].dropna()
                if pv.std() < 0.01 * pv.mean():
                    logger.debug(f"  {path.name}: post_volume has near-zero variance — likely synthetic")
                    continue

            # Align length: take the tail to match market data
            if len(df) > n_rows:
                df = df.tail(n_rows).reset_index(drop=True)
            elif len(df) < n_rows:
                # Pad with zeros at the start
                pad = pd.DataFrame(0, index=range(n_rows - len(df)), columns=df.columns)
                df = pd.concat([pad, df], ignore_index=True)

            logger.info(f"  [REAL SOCIAL] Loaded from {path.name} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"  Could not load {path}: {e}")

    return None


def process_ticker(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process one ticker: market features + social features + labels."""
    df = load_ticker(ticker)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    logger.info(f'Processing {ticker}: {len(df)} rows')

    # Technical indicators
    df = market_extractor.calculate_technical_indicators(df)

    # Sector decoupling
    df = decoupler.enrich_dataframe(df, ticker)

    # ── Social features: try real first, fall back to proxy ───────────────────
    real_social = load_real_social(ticker, len(df))
    social_cols = ['sentiment_score', 'post_volume', 'engagement_score',
                   'bot_activity_score', 'narrative_similarity', 'sector_divergence']

    if real_social is not None:
        # Use real social data
        for col in social_cols:
            if col in real_social.columns:
                df[col] = real_social[col].values[:len(df)]
            else:
                df[col] = 0.0
        logger.info(f"  {ticker}: using REAL social data")
    else:
        # Fall back to market proxy
        social_mat = social_extractor.build_feature_matrix(df)
        for i, col in enumerate(social_cols):
            df[col] = social_mat[:, i] if i < social_mat.shape[1] else 0.0
        logger.info(f"  {ticker}: using MARKET-PROXY social data (no real social found)")

    # ── Labels ────────────────────────────────────────────────────────────────
    labels = labeler.detect_pump_and_dump_pattern(df)

    # Log label confidence distribution for quality awareness
    if 'manipulation_confidence' in labels.columns:
        pos = labels[labels['is_manipulation'] == 1]['manipulation_confidence']
        if len(pos) > 0:
            logger.info(
                f"  {ticker}: label confidence  mean={pos.mean():.3f}  "
                f"min={pos.min():.3f}  max={pos.max():.3f}"
            )

    # Sentiment surge proxy: abnormal_zscore > 1.5
    try:
        labels['has_sentiment_surge'] = (
            df['abnormal_zscore'].abs() > 1.5
        ).astype(int).values[:len(labels)]
    except Exception:
        labels['has_sentiment_surge'] = 0

    logger.info(
        f'  {ticker}: {labels["is_manipulation"].sum()} manipulation samples '
        f'({labels["is_manipulation"].mean()*100:.1f}%)'
    )
    return df, labels


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    SOCIAL_DIR.mkdir(parents=True, exist_ok=True)

    all_features, all_labels = [], []
    tickers_with_real_social = []

    for ticker in TICKERS:
        features_df, labels_df = process_ticker(ticker)
        if features_df.empty:
            continue
        n = min(len(features_df), len(labels_df))
        all_features.append(features_df.iloc[:n])
        all_labels.append(labels_df.iloc[:n])

        # Track which tickers had real social data
        if any((SOCIAL_DIR / f'{ticker}{sfx}').exists()
               for sfx in ['_social.csv', '_finnhub.csv']):
            tickers_with_real_social.append(ticker)

    if not all_features:
        logger.error('No data processed. Did you run 02_collect_historical_data.py?')
        return

    combined_features = pd.concat(all_features, ignore_index=True)
    combined_labels   = pd.concat(all_labels,   ignore_index=True)

    features_path = PROCESSED_DIR / 'combined_features.csv'
    labels_path   = LABELED_DIR   / 'labels.csv'
    combined_features.to_csv(features_path, index=False)
    combined_labels.to_csv(labels_path, index=False)

    logger.info(f'Saved {len(combined_features)} rows -> {features_path}')
    logger.info(f'Saved {len(combined_labels)} rows -> {labels_path}')
    logger.info(
        f'Total manipulation rate: '
        f'{combined_labels["is_manipulation"].mean()*100:.2f}%  '
        f'({combined_labels["is_manipulation"].sum()} positive examples)'
    )
    if tickers_with_real_social:
        logger.info(f'Tickers with REAL social data: {tickers_with_real_social}')
    else:
        logger.info(
            'NOTE: No real social data found for any ticker. '
            'Place {ticker}_social.csv or {ticker}_finnhub.csv in data/raw/social/ '
            'to use real sentiment instead of market proxies.'
        )

    # Fit and save StandardScaler for inference
    market_cols = [c for c in MarketFeatureExtractor.MARKET_FEATURES
                   if c in combined_features.columns]
    if market_cols:
        scaler = StandardScaler()
        scaler.fit(combined_features[market_cols].fillna(0).replace([np.inf, -np.inf], 0))
        Path('models/saved_models').mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, 'models/saved_models/market_scaler.pkl')
        logger.info('Saved market_scaler.pkl')


if __name__ == '__main__':
    main()
