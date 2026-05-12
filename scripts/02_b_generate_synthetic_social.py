import pandas as pd
import numpy as np
from pathlib import Path
import logging
from glob import glob
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_social_data():
    """
    Generate synthetic social data based on market data patterns.
    This creates _social.csv files for training when real historical social data is missing.
    """
    market_files = glob('data/raw/market/*_market.csv')
    
    if not market_files:
        logger.error("No market data found! Run scripts/02_collect_historical_data.py first.")
        return

    Path('data/raw/social').mkdir(parents=True, exist_ok=True)
    
    # Load SPY for sector baseline if available
    spy_df = pd.DataFrame()
    if os.path.exists('data/raw/market/SPY_market.csv'):
        try:
            spy_df = pd.read_csv('data/raw/market/SPY_market.csv')
            # Ensure timestamp logic works (assuming string match or conversion needed)
            # For simplicity, we'll reindex by index if lengths match, or merge later
            # Let's use simple return calculation aligned by index for now if close enough
            spy_df['returns'] = spy_df['Close'].pct_change().fillna(0)
        except Exception as e:
            logger.warning(f"Could not load SPY baseline: {e}")
            
    for market_file in market_files:
        try:
            ticker = os.path.basename(market_file).replace('_market.csv', '')
            if ticker in ['SPY', 'QQQ']: continue  # Don't generate social for ETFs
            
            logger.info(f"Generating synthetic social data for {ticker}...")
            
            # Load market data
            df = pd.read_csv(market_file)
            if df.empty:
                logger.warning(f"Skipping empty market file for {ticker}")
                continue

                
            # Create social DataFrame aligned with market timestamps
            social_df = pd.DataFrame()
            social_df['timestamp'] = df['timestamp']
            social_df['ticker'] = ticker
            
            # --- Generate Synthetic Features based on Market Moves ---
            
            # 1. Sentiment Score (-1 to 1) 
            # correlate with RSI (if available) or Returns
            if 'RSI' in df.columns:
                # RSI > 70 -> High Sentiment (0.5 to 1.0)
                # RSI < 30 -> Low Sentiment (-1.0 to -0.5)
                rsi_norm = (df['RSI'] - 50) / 50  # -1 to 1
                noise = np.random.normal(0, 0.2, len(df))
                social_df['sentiment_score'] = np.clip(rsi_norm + noise, -1, 1)
            else:
                # Use returns if no RSI
                returns = df['Close'].pct_change().fillna(0)
                social_df['sentiment_score'] = np.clip(returns * 10 + np.random.normal(0, 0.3, len(df)), -1, 1)

            # 2. Post Volume (normalized 0 to 100)
            # Correlate with Market Volume
            if 'Volume' in df.columns:
                vol_norm = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-9)
                vol_norm = vol_norm.fillna(0)
                # High volume -> High post volume
                base_volume = np.abs(vol_norm) * 20 + 10 
                # Ensure non-negative and finite
                base_volume = np.nan_to_num(base_volume, nan=10.0, posinf=500.0, neginf=0.0)
                base_volume = np.clip(base_volume, 0, 400)
                
                social_df['post_volume'] = (base_volume + np.random.poisson(5, len(df))).astype(int)
            else:
                social_df['post_volume'] = np.random.poisson(10, len(df))

            # 3. Engagement Score (0 to 1000)
            # Correlate with Volatility (High volatility -> viral posts)
            if 'Volatility_20h' in df.columns:
                volatility = df['Volatility_20h'].fillna(0)
                social_df['engagement_score'] = (volatility * 10000) + social_df['post_volume'] * 2
            else:
                social_df['engagement_score'] = social_df['post_volume'] * np.random.uniform(1, 5, len(df))
                
            social_df['engagement_score'] = social_df['engagement_score'].fillna(0).replace([np.inf, -np.inf], 0)
            social_df['engagement_score'] = social_df['engagement_score'].abs().astype(int)

            # --- Advanced Final Year Project Features ---
            
            # 1. Sector Divergence (Alpha)
            if not spy_df.empty and len(spy_df) >= len(df):
                # Simple alignment by taking tail (recent data)
                # In real app, merge on timestamp. Here we assume aligned hourly data collection
                spy_subset = spy_df.tail(len(df)).reset_index(drop=True)
                stock_ret = df['Close'].pct_change().fillna(0)
                spy_ret = spy_subset['returns'].fillna(0) if 'returns' in spy_subset else np.zeros(len(df))
                social_df['sector_divergence'] = (stock_ret - spy_ret)
            else:
                social_df['sector_divergence'] = np.random.normal(0, 0.01, len(df)) # Fallback
                
            # 2. Bot Activity Score (0-1)
            # Bots are active during high volume, low sentiment-diversity times
            # Simulate: High volume z-score -> higher bot prob
            vol_z = social_df['post_volume'] / (social_df['post_volume'].mean() + 1e-9)
            bot_noise = np.random.beta(2, 5, len(df)) # Skewed towards 0
            social_df['bot_activity_score'] = np.clip(vol_z * 0.1 + bot_noise, 0, 1)
            
            # 3. FinBERT Sentiment (Confidence weighted)
            # Use sentiment_score but add 'confidence' dimension
            social_df['finbert_sentiment'] = social_df['sentiment_score']
            social_df['finbert_confidence'] = np.abs(social_df['sentiment_score']) * 0.8 + np.random.uniform(0, 0.2, len(df))
            
            # 4. Narrative Consensus (0-1)
            # High consensus = coordinated campaign?
            social_df['narrative_similarity'] = np.abs(social_df['sentiment_score']) * 0.6 + np.random.uniform(0, 0.4, len(df))

            # Save
            output_file = f'data/raw/social/{ticker}_social.csv'
            social_df.to_csv(output_file, index=False)
            logger.info(f"   Saved {len(social_df)} records to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate social data for {ticker}: {e}")

if __name__ == "__main__":
    generate_synthetic_social_data()
