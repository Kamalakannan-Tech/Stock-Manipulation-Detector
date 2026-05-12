"""
Main service orchestrator that manages all background services.
"""
import asyncio
import logging
import signal
import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_collection.live_market_stream import LiveMarketStream
from src.inference.realtime_predictor import RealtimePredictor
from src.inference.alert_manager import AlertManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceOrchestrator:
    """
    Orchestrates all background services for the stock manipulation detection system.
    """
    
    def __init__(self):
        """Initialize the service orchestrator."""
        self.services = {}
        self.running = False
        
        # Get configuration from environment
        self.monitored_tickers = os.getenv('MONITORED_TICKERS', 'GME,AMC,TSLA,AAPL,MSFT,NVDA').split(',')
        
        logger.info(f"Service orchestrator initialized for tickers: {', '.join(self.monitored_tickers)}")
    
    def initialize_services(self):
        """Initialize all services."""
        logger.info("Initializing services...")
        
        try:
            # Initialize predictor
            self.predictor = RealtimePredictor(
                model_path=os.getenv('MODEL_PATH', 'models/saved_models/best_model.pth'),
                device=os.getenv('DEVICE', 'cpu')
            )
            logger.info("✓ Predictor initialized")
            
            # Initialize alert manager
            self.alert_manager = AlertManager()
            logger.info("✓ Alert manager initialized")
            
            # Initialize market stream
            self.market_stream = LiveMarketStream(
                api_key=os.getenv('ALPACA_API_KEY'),
                secret_key=os.getenv('ALPACA_SECRET_KEY'),
                base_url=os.getenv('ALPACA_BASE_URL', 'https://api.alpaca.markets')
            )
            
            # Subscribe to tickers
            self.market_stream.subscribe_tickers(self.monitored_tickers)
            
            # Add callback to make predictions on new trades
            self.market_stream.add_trade_callback(self.on_trade_update)
            logger.info("✓ Market stream initialized")
            
            logger.info("All services initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing services: {e}", exc_info=True)
            raise
    
    def on_trade_update(self, trade_data):
        """
        Callback for market trade updates.
        Makes predictions periodically (not on every trade to avoid overload).
        """
        ticker = trade_data.get('ticker')
        
        # Make prediction every 100 trades (to avoid overload)
        # In production, you might use a time-based trigger instead
        if hash(trade_data.get('timestamp', '')) % 100 == 0:
            try:
                prediction = self.predictor.predict(ticker)
                
                if prediction and prediction.get('risk_score', 0) >= 50:
                    # Create alert for high risk
                    alert = self.alert_manager.create_alert(ticker, prediction)
                    if alert:
                        logger.warning(f"⚠️  ALERT: {alert['message']}")
            except Exception as e:
                logger.error(f"Error making prediction: {e}")
    
    async def start_market_stream(self):
        """Start the market data stream with exponential backoff (max 5 retries)."""
        MAX_RETRIES = 5
        retry = 0
        delay = 5  # seconds

        while self.running and retry < MAX_RETRIES:
            try:
                logger.info(f"Connecting to Alpaca stream (attempt {retry + 1}/{MAX_RETRIES})...")
                await self.market_stream.start()
                retry = 0  # reset on clean run
            except ValueError as e:
                if 'connection limit' in str(e).lower():
                    retry += 1
                    wait = delay * (2 ** (retry - 1))  # 5, 10, 20, 40, 80 s
                    logger.warning(
                        f"Alpaca connection limit exceeded (attempt {retry}/{MAX_RETRIES}). "
                        f"Waiting {wait}s before retry..."
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Alpaca auth error: {e}")
                    break
            except Exception as e:
                retry += 1
                wait = delay * (2 ** (retry - 1))
                logger.error(f"Market stream error (attempt {retry}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)

        if retry >= MAX_RETRIES:
            logger.warning(
                "Alpaca stream: max retries reached. "
                "Streaming disabled — polling loop continues using yfinance."
            )

    async def poll_tickers(self):
        """
        Continuous live monitoring loop.
        Runs model inference on every monitored ticker every POLL_INTERVAL_SECONDS.
        Operates independently of Alpaca WebSocket using fresh yfinance data.
        """
        POLL_INTERVAL = int(os.getenv('POLL_INTERVAL_SECONDS', '300'))  # default 5 min
        logger.info(f"[POLL] Starting live monitoring loop (every {POLL_INTERVAL}s)")

        while self.running:
            for ticker in self.monitored_tickers:
                ticker = ticker.strip()
                try:
                    prediction = self.predictor.predict(ticker)
                    if prediction:
                        risk  = prediction.get('risk_score', 0)
                        level = prediction.get('risk_level', 'unknown').upper()
                        prob  = prediction.get('manipulation_probability', 0)
                        logger.info(
                            f"[POLL] {ticker} risk={level} ({risk:.1f}%) prob={prob:.3f}"
                        )
                        if risk >= 50:
                            alert = self.alert_manager.create_alert(ticker, prediction)
                            if alert:
                                logger.warning(
                                    f"[POLL] ALERT {ticker}: {alert.get('message', '')}"
                                )
                except Exception as e:
                    logger.error(f"[POLL] Error on {ticker}: {e}")
                await asyncio.sleep(2)   # rate-limit yfinance calls

            logger.info(f"[POLL] Scan complete - next scan in {POLL_INTERVAL}s")
            await asyncio.sleep(POLL_INTERVAL)
    
    async def run(self):
        """Run all services."""
        self.running = True
        logger.info("="*60)
        logger.info("Starting Stock Manipulation Detection System")
        logger.info("="*60)
        
        # Initialize services
        self.initialize_services()
        
        # Run Alpaca stream + continuous polling loop concurrently
        tasks = [
            asyncio.create_task(self.start_market_stream()),
            asyncio.create_task(self.poll_tickers()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Services cancelled")
        except Exception as e:
            logger.error(f"Error running services: {e}", exc_info=True)
    
    def stop(self):
        """Stop all services."""
        logger.info("Stopping services...")
        self.running = False
        
        if hasattr(self, 'market_stream'):
            self.market_stream.stop()
        
        if hasattr(self, 'predictor'):
            self.predictor.close()
        
        if hasattr(self, 'alert_manager'):
            self.alert_manager.close()
        
        logger.info("All services stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


async def main():
    """Main entry point."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run orchestrator
    orchestrator = ServiceOrchestrator()
    
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
    finally:
        orchestrator.stop()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nExiting...")
