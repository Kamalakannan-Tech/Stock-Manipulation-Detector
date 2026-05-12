"""
Real-time market data streaming using Alpaca WebSocket API.
Streams live trades and quotes for monitored tickers.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Callable, Dict
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from alpaca_trade_api.stream import Stream
    from alpaca_trade_api.common import URL
except ImportError:
    print("Warning: alpaca-trade-api not installed. Install with: pip install alpaca-trade-api")
    Stream = None

import pandas as pd
from pymongo import MongoClient

# Import cache fallback instead of direct Redis
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.cache_fallback import get_redis_client

logger = logging.getLogger(__name__)

class LiveMarketStream:
    """
    Real-time market data streaming service using Alpaca WebSocket.
    Streams trades and quotes, calculates technical indicators in real-time.
    """
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = 'https://paper-api.alpaca.markets',
                 mongo_uri: str = 'mongodb://localhost:27017/',
                 mongo_db: str = 'stock_manipulation',
                 redis_url: str = 'redis://localhost:6379'):
        """
        Initialize live market data stream.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            base_url: Alpaca base URL (paper or live)
            mongo_uri: MongoDB connection URI
            mongo_db: MongoDB database name
            redis_url: Redis connection URL
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        
        # Initialize database connections
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[mongo_db]
        self.market_collection = self.db['live_market_data']
        
        # Initialize Redis/cache with fallback
        self.redis_client = get_redis_client(redis_url, decode_responses=True)
        
        # Initialize Alpaca stream
        if Stream:
            self.stream = Stream(
                api_key,
                secret_key,
                base_url=URL(base_url),
                data_feed='iex'  # Use IEX for real-time data
            )
        else:
            self.stream = None
            logger.warning("Alpaca Stream not available")
        
        # Track subscribed tickers
        self.subscribed_tickers = set()
        
        # Data buffers for technical indicators
        self.price_buffers = {}  # ticker -> list of recent prices
        self.volume_buffers = {}  # ticker -> list of recent volumes
        self.buffer_size = 100  # Keep last 100 data points
        
        # Callbacks for data processing
        self.trade_callbacks = []
        self.quote_callbacks = []
        
        logger.info("Live market stream initialized")
    
    async def on_trade(self, trade):
        """Handle incoming trade data."""
        try:
            ticker = trade.symbol
            
            # Extract trade data
            trade_data = {
                'ticker': ticker,
                'timestamp': trade.timestamp,
                'price': float(trade.price),
                'size': int(trade.size),
                'exchange': trade.exchange,
                'conditions': trade.conditions,
                'data_type': 'trade'
            }
            
            # Update price buffer
            if ticker not in self.price_buffers:
                self.price_buffers[ticker] = []
                self.volume_buffers[ticker] = []
            
            self.price_buffers[ticker].append(trade_data['price'])
            self.volume_buffers[ticker].append(trade_data['size'])
            
            # Trim buffers
            if len(self.price_buffers[ticker]) > self.buffer_size:
                self.price_buffers[ticker] = self.price_buffers[ticker][-self.buffer_size:]
                self.volume_buffers[ticker] = self.volume_buffers[ticker][-self.buffer_size:]
            
            # Calculate real-time indicators
            indicators = self._calculate_realtime_indicators(ticker)
            trade_data.update(indicators)
            
            # Store in MongoDB
            self.market_collection.insert_one(trade_data)
            
            # Cache latest data in Redis
            redis_key = f"market:live:{ticker}"
            self.redis_client.setex(
                redis_key,
                300,  # 5 minute expiry
                json.dumps(trade_data, default=str)
            )
            
            # Publish to Redis pub/sub for real-time updates
            self.redis_client.publish(
                f"market_updates:{ticker}",
                json.dumps(trade_data, default=str)
            )
            
            # Execute callbacks
            for callback in self.trade_callbacks:
                try:
                    callback(trade_data)
                except Exception as e:
                    logger.error(f"Error in trade callback: {e}")
            
            logger.debug(f"Trade: {ticker} @ ${trade_data['price']:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing trade: {e}")
    
    async def on_quote(self, quote):
        """Handle incoming quote data."""
        try:
            ticker = quote.symbol
            
            quote_data = {
                'ticker': ticker,
                'timestamp': quote.timestamp,
                'bid_price': float(quote.bid_price),
                'bid_size': int(quote.bid_size),
                'ask_price': float(quote.ask_price),
                'ask_size': int(quote.ask_size),
                'bid_exchange': quote.bid_exchange,
                'ask_exchange': quote.ask_exchange,
                'data_type': 'quote'
            }
            
            # Calculate spread
            quote_data['spread'] = quote_data['ask_price'] - quote_data['bid_price']
            quote_data['spread_pct'] = (quote_data['spread'] / quote_data['ask_price']) * 100 if quote_data['ask_price'] > 0 else 0
            
            # Cache in Redis
            redis_key = f"market:quote:{ticker}"
            self.redis_client.setex(
                redis_key,
                60,  # 1 minute expiry
                json.dumps(quote_data, default=str)
            )
            
            # Execute callbacks
            for callback in self.quote_callbacks:
                try:
                    callback(quote_data)
                except Exception as e:
                    logger.error(f"Error in quote callback: {e}")
            
        except Exception as e:
            logger.error(f"Error processing quote: {e}")
    
    def _calculate_realtime_indicators(self, ticker: str) -> Dict:
        """Calculate technical indicators from buffered data."""
        indicators = {}
        
        if ticker not in self.price_buffers or len(self.price_buffers[ticker]) < 2:
            return indicators
        
        prices = self.price_buffers[ticker]
        volumes = self.volume_buffers[ticker]
        
        # Current price
        current_price = prices[-1]
        indicators['current_price'] = current_price
        
        # Returns
        if len(prices) >= 2:
            indicators['return_1tick'] = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] > 0 else 0
        
        if len(prices) >= 5:
            indicators['return_5tick'] = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
        
        # Moving averages
        if len(prices) >= 5:
            indicators['sma_5'] = sum(prices[-5:]) / 5
        
        if len(prices) >= 20:
            indicators['sma_20'] = sum(prices[-20:]) / 20
        
        # Volatility (std dev of returns)
        if len(prices) >= 10:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            mean_return = sum(returns[-10:]) / 10
            variance = sum((r - mean_return) ** 2 for r in returns[-10:]) / 10
            indicators['volatility_10tick'] = variance ** 0.5
        
        # Volume metrics
        if len(volumes) >= 20:
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            indicators['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1
        
        return indicators
    
    def subscribe_tickers(self, tickers: List[str]):
        """
        Subscribe to real-time data for specified tickers.
        
        Args:
            tickers: List of ticker symbols
        """
        if not self.stream:
            logger.error("Stream not initialized")
            return
        
        for ticker in tickers:
            if ticker not in self.subscribed_tickers:
                self.subscribed_tickers.add(ticker)
                logger.info(f"Subscribing to {ticker}")
        
        # Subscribe to trades and quotes
        self.stream.subscribe_trades(self.on_trade, *list(self.subscribed_tickers))
        self.stream.subscribe_quotes(self.on_quote, *list(self.subscribed_tickers))
    
    def add_trade_callback(self, callback: Callable):
        """Add a callback function to be called on each trade."""
        self.trade_callbacks.append(callback)
    
    def add_quote_callback(self, callback: Callable):
        """Add a callback function to be called on each quote."""
        self.quote_callbacks.append(callback)
    
    async def start(self):
        """Start the streaming service."""
        if not self.stream:
            logger.error("Cannot start stream - not initialized")
            return
        
        logger.info(f"Starting live market stream for {len(self.subscribed_tickers)} tickers")
        logger.info(f"Subscribed tickers: {', '.join(self.subscribed_tickers)}")
        
        try:
            await self.stream._run_forever()
        except Exception as e:
            logger.error(f"Stream error: {e}")
    
    def stop(self):
        """Stop the streaming service."""
        if self.stream:
            logger.info("Stopping live market stream")
            self.stream.stop()
        
        # Close connections
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            self.redis_client.close()
    
    def get_latest_data(self, ticker: str) -> Dict:
        """Get latest cached data for a ticker from Redis."""
        redis_key = f"market:live:{ticker}"
        data = self.redis_client.get(redis_key)
        
        if data:
            return json.loads(data)
        return {}


async def main():
    """Test the live market stream."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize stream
    stream = LiveMarketStream(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    )
    
    # Add a simple callback to print trades
    def print_trade(trade_data):
        print(f"[{trade_data['timestamp']}] {trade_data['ticker']}: ${trade_data['price']:.2f} "
              f"(size: {trade_data['size']}, return: {trade_data.get('return_1tick', 0):.4f})")
    
    stream.add_trade_callback(print_trade)
    
    # Subscribe to tickers
    tickers = os.getenv('MONITORED_TICKERS', 'GME,AMC,TSLA').split(',')
    stream.subscribe_tickers(tickers)
    
    # Start streaming
    try:
        await stream.start()
    except KeyboardInterrupt:
        print("\nStopping stream...")
        stream.stop()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())
