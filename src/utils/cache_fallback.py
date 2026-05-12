"""
Simple in-memory cache as Redis fallback.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)

class InMemoryCache:
    """Simple in-memory cache to replace Redis when unavailable."""
    
    def __init__(self):
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.pubsub_subscribers: Dict[str, list] = {}
        logger.info("Using in-memory cache (Redis not available)")
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if key in self.cache:
            value, expiry = self.cache[key]
            if expiry is None or datetime.now() < expiry:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: str):
        """Set value in cache without expiry."""
        self.cache[key] = (value, None)
    
    def setex(self, key: str, seconds: int, value: str):
        """Set value with expiry."""
        expiry = datetime.now() + timedelta(seconds=seconds)
        self.cache[key] = (value, expiry)
    
    def publish(self, channel: str, message: str):
        """Publish message (no-op for in-memory)."""
        logger.debug(f"Published to {channel}: {message[:100]}")
    
    def close(self):
        """Close connection (no-op)."""
        pass

def get_redis_client(redis_url: str = 'redis://localhost:6379', decode_responses: bool = True):
    """
    Get Redis client, falling back to in-memory cache if Redis unavailable.
    """
    try:
        import redis
        client = redis.from_url(redis_url, decode_responses=decode_responses)
        # Test connection
        client.ping()
        logger.info("Connected to Redis")
        return client
    except Exception as e:
        logger.warning(f"Redis not available ({e}), using in-memory cache")
        return InMemoryCache()
