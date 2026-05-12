import sys
sys.path.append('.')
import yfinance as yf
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtendedMarketCollector:
    """Collect extensive historical data without API keys"""
    
    def collect_historical_data(self, ticker, months_back=6):
        """
        Collect historical intraday data
        
        Args:
            ticker: Stock symbol
            months_back: How many months of history to collect
        """
        logger.info(f"Collecting {months_back} months of data for {ticker}...")
        
        all_data = []
        
        # yfinance limits: max 60 days per request for 1-minute data
        # So we'll collect in chunks
        end_date = datetime.now()
        
        # Calculate number of 30-day chunks needed
        chunks = months_back  # 1 chunk = ~30 days
        
        for i in range(chunks):
            try:
                # Calculate date range for this chunk
                chunk_end = end_date - timedelta(days=i*30)
                chunk_start = chunk_end - timedelta(days=30)
                
                logger.info(f"  Chunk {i+1}/{chunks}: {chunk_start.date()} to {chunk_end.date()}")
                
                # Download data
                stock = yf.Ticker(ticker)
                df = stock.history(
                    start=chunk_start.strftime('%Y-%m-%d'),
                    end=chunk_end.strftime('%Y-%m-%d'),
                    interval='1h'  # Use hourly data for longer history
                )
                
                if not df.empty:
                    df = df.reset_index()
                    df['ticker'] = ticker
                    all_data.append(df)
                    logger.info(f"    ✓ Got {len(df)} records")
                else:
                    logger.warning(f"    ✗ No data for this period")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"    ✗ Error: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all chunks
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values('Datetime' if 'Datetime' in combined.columns else 'Date')
        combined = combined.drop_duplicates()
        
        logger.info(f"✓ Total records for {ticker}: {len(combined)}")
        return combined
    
    def calculate_features(self, df):
        """Calculate technical indicators"""
        
        if df.empty:
            return df
        
        try:
            # Rename datetime column
            if 'Datetime' in df.columns:
                df = df.rename(columns={'Datetime': 'timestamp'})
            elif 'Date' in df.columns:
                df = df.rename(columns={'Date': 'timestamp'})
            
            # Price features
            df['Returns_1h'] = df['Close'].pct_change(1)
            df['Returns_4h'] = df['Close'].pct_change(4)
            df['Returns_1d'] = df['Close'].pct_change(24)
            
            # Moving averages
            df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
            df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            
            # Volatility
            df['Volatility_5h'] = df['Close'].pct_change().rolling(window=5, min_periods=1).std()
            df['Volatility_20h'] = df['Close'].pct_change().rolling(window=20, min_periods=1).std()
            
            # Volume features
            df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            df['Volume_Std'] = df['Volume'].rolling(window=20, min_periods=1).std()
            df['Abnormal_Volume_Zscore'] = (df['Volume'] - df['Volume_MA']) / (df['Volume_Std'] + 1e-10)
            
            # Price momentum
            df['Price_Momentum_5h'] = df['Close'].pct_change(periods=5)
            df['Price_Momentum_20h'] = df['Close'].pct_change(periods=20)
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['BB_Std'] = df['Close'].rolling(window=20, min_periods=1).std()
            df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
            df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            
            # RSI (Relative Strength Index)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # High-Low spread
            df['HL_Spread'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
            
            # Price position in range
            df['Price_Position'] = (df['Close'] - df['Low']) / ((df['High'] - df['Low']) + 1e-10)
            
            # Fill NaN values
            df = df.ffill().bfill().fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return df


def collect_all_tickers(tickers, months_back=6):
    """Collect data for all tickers"""
    
    collector = ExtendedMarketCollector()
    
    # Ensure directory exists
    Path('data/raw/market').mkdir(parents=True, exist_ok=True)
    
    for ticker in tickers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {ticker}")
        logger.info(f"{'='*60}")
        
        try:
            # Collect raw data
            df = collector.collect_historical_data(ticker, months_back=months_back)
            
            if df.empty:
                logger.warning(f"No data collected for {ticker}")
                continue
            
            # Calculate features
            df = collector.calculate_features(df)
            
            # Save to CSV
            filename = f'data/raw/market/{ticker}_market.csv'
            df.to_csv(filename, index=False)
            logger.info(f"✓ Saved to {filename}")
            logger.info(f"  Records: {len(df)}")
            logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {ticker}: {e}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info("Data collection complete!")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect historical market data')
    parser.add_argument('--tickers', nargs='+', 
                       default=['GME','AMC','TSLA','AAPL','MSFT','NVDA','BBBY','RIVN'],
                       help='List of ticker symbols')
    parser.add_argument('--months', type=int, default=6,
                       help='Months of historical data to collect')
    
    args = parser.parse_args()
    
    logger.info("Starting extended data collection...")
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"History: {args.months} months")
    
    collect_all_tickers(args.tickers, months_back=args.months)