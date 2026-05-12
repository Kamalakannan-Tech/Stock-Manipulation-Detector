import sys
sys.path.append('.')
from pymongo import MongoClient
from config.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    client = MongoClient(config.MONGODB_URI)
    db = client[config.MONGODB_DATABASE]
    
    collections = ['social_twitter', 'social_reddit', 'social_stocktwits', 'market_data', 'alerts']
    
    for collection in collections:
        if collection not in db.list_collection_names():
            db.create_collection(collection)
            logger.info(f"Created collection: {collection}")
    
    logger.info("Database setup complete!")

if __name__ == '__main__':
    setup_database()
