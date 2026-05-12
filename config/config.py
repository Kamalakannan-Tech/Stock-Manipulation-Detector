import os
from dotenv import load_dotenv
import yaml
from pathlib import Path

load_dotenv()

class Config:
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODEL_DIR = BASE_DIR / 'models'
    LOG_DIR = BASE_DIR / 'logs'
    
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'stock_manipulation')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    STOCKTWITS_TOKEN = os.getenv('STOCKTWITS_TOKEN')
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'StockManipulationDetector/1.0')
    
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/saved_models/best_model.pth')
    DEVICE = os.getenv('DEVICE', 'cuda')
    
    MONITORED_TICKERS = os.getenv('MONITORED_TICKERS', 'GME,AMC,BBBY,TSLA,NVDA').split(',')
    
    SOCIAL_FEATURES = 6
    MARKET_FEATURES = 13
    
    @classmethod
    def load_model_config(cls):
        config_path = cls.BASE_DIR / 'config' / 'model_config.yaml'
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

config = Config()
