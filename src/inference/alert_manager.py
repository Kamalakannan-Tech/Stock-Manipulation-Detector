"""
Alert management system for stock manipulation detection.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from enum import Enum
import json
import os
import sys

# pymongo is optional — alert persistence is skipped gracefully when absent
try:
    from pymongo import MongoClient
    _PYMONGO_AVAILABLE = True
except ImportError:
    MongoClient = None          # type: ignore
    _PYMONGO_AVAILABLE = False

# Add project root to path (insert at front so src.* is always found first)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.cache_fallback import get_redis_client

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertManager:
    """
    Manages alert generation, storage, and retrieval.
    Prevents duplicate alerts and provides alert statistics.
    """
    
    def __init__(self,
                 mongo_uri='mongodb://localhost:27017/',
                 mongo_db='stock_manipulation',
                 redis_url='redis://localhost:6379',
                 dedup_window_minutes=60):
        """
        Initialize alert manager.
        
        Args:
            mongo_uri: MongoDB connection URI
            mongo_db: MongoDB database name
            redis_url: Redis connection URL
            dedup_window_minutes: Time window for deduplication
        """
        # Database connections (optional — graceful fallback when MongoDB unavailable)
        self.mongo_client = None
        self.db = None
        self.alerts_collection = None
        if _PYMONGO_AVAILABLE and MongoClient is not None:
            try:
                self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
                self.mongo_client.admin.command('ping')
                self.db = self.mongo_client[mongo_db]
                self.alerts_collection = self.db['alerts']
                # Create indexes
                self.alerts_collection.create_index([('ticker', 1), ('timestamp', -1)])
                self.alerts_collection.create_index([('severity', 1)])
                self.alerts_collection.create_index([('acknowledged', 1)])
                logger.info("AlertManager: MongoDB connected")
            except Exception as e:
                logger.warning(f"AlertManager: MongoDB unavailable ({e}) — alerts stored in memory only")
                self.mongo_client = None
                self.db = None
                self.alerts_collection = None

        # In-memory fallback alert log (used when MongoDB is unavailable)
        self._mem_alerts: list = []

        # Redis for caching (with fallback to in-memory)
        self.redis_client = get_redis_client(redis_url, decode_responses=True)

        # Configuration
        self.dedup_window = timedelta(minutes=dedup_window_minutes)

        logger.info("Alert manager initialized")
    
    def create_alert(self, ticker: str, prediction: Dict, 
                    force: bool = False) -> Optional[Dict]:
        """
        Create an alert based on prediction results.
        
        Args:
            ticker: Stock ticker symbol
            prediction: Prediction dictionary from RealtimePredictor
            force: Force alert creation even if duplicate
            
        Returns:
            Created alert dictionary or None if deduplicated
        """
        try:
            risk_level = prediction.get('risk_level', 'low')
            risk_score = prediction.get('risk_score', 0)
            
            # Determine severity
            severity = self._risk_to_severity(risk_level)
            
            # Check for duplicate alerts (unless forced)
            if not force and self._is_duplicate(ticker, severity):
                logger.debug(f"Skipping duplicate alert for {ticker}")
                return None
            
            # Create alert
            alert = {
                'ticker': ticker,
                'timestamp': datetime.now(),
                'severity': severity.value,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'manipulation_probability': prediction.get('manipulation_probability', 0),
                'volume_anomaly': prediction.get('volume_anomaly_prob', 0),
                'sentiment_surge': prediction.get('sentiment_surge_prob', 0),
                'lead_lag_score': prediction.get('lead_lag_score', 0),
                'message': self._generate_message(ticker, prediction),
                'acknowledged': False,
                'acknowledged_at': None,
                'acknowledged_by': None,
                'model_version': prediction.get('model_version', 'unknown')
            }
            
            # Store in MongoDB (if available) or in-memory fallback
            if self.alerts_collection is not None:
                result = self.alerts_collection.insert_one(alert)
                alert['_id'] = result.inserted_id
            else:
                self._mem_alerts.append(alert)
                # Trim to last 500
                self._mem_alerts = self._mem_alerts[-500:]

            # Cache in Redis
            redis_key = f"alert:latest:{ticker}"
            self.redis_client.setex(
                redis_key,
                3600,  # 1 hour expiry
                json.dumps(alert, default=str)
            )
            
            # Publish to Redis pub/sub
            self.redis_client.publish(
                'alerts',
                json.dumps(alert, default=str)
            )
            
            logger.info(f"Created {severity.value} alert for {ticker}: {alert['message']}")
            
            return alert
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}", exc_info=True)
            return None
    
    def _risk_to_severity(self, risk_level: str) -> AlertSeverity:
        """Convert risk level to alert severity."""
        mapping = {
            'critical': AlertSeverity.CRITICAL,
            'high': AlertSeverity.HIGH,
            'medium': AlertSeverity.MEDIUM,
            'low': AlertSeverity.LOW
        }
        return mapping.get(risk_level.lower(), AlertSeverity.LOW)
    
    def _is_duplicate(self, ticker: str, severity: AlertSeverity) -> bool:
        """Check if a similar alert was recently created."""
        cutoff_time = datetime.now() - self.dedup_window

        if self.alerts_collection is not None:
            existing = self.alerts_collection.find_one({
                'ticker': ticker,
                'severity': severity.value,
                'timestamp': {'$gte': cutoff_time}
            })
            return existing is not None

        # Fallback: search in-memory list
        for a in self._mem_alerts:
            if (a.get('ticker') == ticker
                    and a.get('severity') == severity.value
                    and a.get('timestamp', datetime.min) >= cutoff_time):
                return True
        return False
    
    def _generate_message(self, ticker: str, prediction: Dict) -> str:
        """Generate human-readable alert message."""
        risk_score = prediction.get('risk_score', 0)
        manip_prob = prediction.get('manipulation_probability', 0)
        
        if risk_score >= 75:
            return f"CRITICAL: ${ticker} shows strong manipulation signals ({risk_score:.0f}% risk, {manip_prob:.1%} probability)"
        elif risk_score >= 50:
            return f"HIGH RISK: ${ticker} exhibits suspicious activity ({risk_score:.0f}% risk, {manip_prob:.1%} probability)"
        elif risk_score >= 25:
            return f"MODERATE: ${ticker} shows elevated risk indicators ({risk_score:.0f}% risk)"
        else:
            return f"LOW: ${ticker} risk within normal range ({risk_score:.0f}%)"
    
    def get_alerts(self, 
                   ticker: Optional[str] = None,
                   severity: Optional[str] = None,
                   acknowledged: Optional[bool] = None,
                   limit: int = 100,
                   skip: int = 0) -> List[Dict]:
        """
        Get alerts with filtering.
        
        Args:
            ticker: Filter by ticker (optional)
            severity: Filter by severity (optional)
            acknowledged: Filter by acknowledgment status (optional)
            limit: Maximum number of alerts to return
            skip: Number of alerts to skip (pagination)
            
        Returns:
            List of alert dictionaries
        """
        query = {}
        
        if ticker:
            query['ticker'] = ticker
        
        if severity:
            query['severity'] = severity
        
        if acknowledged is not None:
            query['acknowledged'] = acknowledged
        
        if self.alerts_collection is not None:
            alerts = self.alerts_collection.find(query).sort(
                'timestamp', -1
            ).skip(skip).limit(limit)
            return list(alerts)

        # Fallback: filter in-memory list
        results = [
            a for a in self._mem_alerts
            if all(a.get(k) == v for k, v in query.items())
        ]
        results.sort(key=lambda a: a.get('timestamp', datetime.min), reverse=True)
        return results[skip: skip + limit]
    
    def get_alert_by_id(self, alert_id: str) -> Optional[Dict]:
        """Get a specific alert by ID."""
        if self.alerts_collection is None:
            # Fallback: search memory by string repr of _id
            for a in self._mem_alerts:
                if str(a.get('_id', '')) == alert_id:
                    return a
            return None
        try:
            from bson.objectid import ObjectId
            alert = self.alerts_collection.find_one({'_id': ObjectId(alert_id)})
            return alert
        except Exception as e:
            logger.error(f"Error getting alert: {e}")
            return None
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = 'system') -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID
            acknowledged_by: Who acknowledged the alert

        Returns:
            True if successful
        """
        if self.alerts_collection is None:
            # Fallback: update in-memory
            for a in self._mem_alerts:
                if str(a.get('_id', '')) == alert_id:
                    a['acknowledged']    = True
                    a['acknowledged_at'] = datetime.now()
                    a['acknowledged_by'] = acknowledged_by
                    return True
            return False
        try:
            from bson.objectid import ObjectId
            result = self.alerts_collection.update_one(
                {'_id': ObjectId(alert_id)},
                {
                    '$set': {
                        'acknowledged':    True,
                        'acknowledged_at': datetime.now(),
                        'acknowledged_by': acknowledged_by
                    }
                }
            )
            if result.modified_count > 0:
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def get_alert_stats(self, hours: int = 24) -> Dict:
        """
        Get alert statistics for the specified time period.

        Args:
            hours: Time period in hours

        Returns:
            Dictionary with statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # In-memory fallback when MongoDB is not available
        if self.alerts_collection is None:
            alerts = [
                a for a in self._mem_alerts
                if a.get('timestamp', datetime.min) >= cutoff_time
            ]
        else:
            try:
                alerts = list(self.alerts_collection.find(
                    {'timestamp': {'$gte': cutoff_time}}
                ))
            except Exception as e:
                logger.error(f"get_alert_stats MongoDB error: {e}")
                alerts = []

        if not alerts:
            return {
                'total_alerts':         0,
                'by_severity':          {},
                'by_ticker':            {},
                'acknowledged_count':   0,
                'unacknowledged_count': 0,
            }

        stats = {
            'total_alerts':         len(alerts),
            'by_severity':          {},
            'by_ticker':            {},
            'acknowledged_count':   sum(1 for a in alerts if a.get('acknowledged', False)),
            'unacknowledged_count': sum(1 for a in alerts if not a.get('acknowledged', False)),
        }
        for alert in alerts:
            sev = alert.get('severity', 'unknown')
            stats['by_severity'][sev] = stats['by_severity'].get(sev, 0) + 1
            tkr = alert.get('ticker', 'unknown')
            stats['by_ticker'][tkr] = stats['by_ticker'].get(tkr, 0) + 1

        return stats
    
    def get_latest_alert(self, ticker: str) -> Optional[Dict]:
        """Get the latest alert for a ticker."""
        if self.alerts_collection is not None:
            return self.alerts_collection.find_one(
                {'ticker': ticker}, sort=[('timestamp', -1)]
            )
        # Fallback: search in-memory
        candidates = [a for a in self._mem_alerts if a.get('ticker') == ticker]
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.get('timestamp', datetime.min))
    
    def close(self):
        """Close database connections."""
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            self.redis_client.close()


if __name__ == '__main__':
    # Test the alert manager
    
    logging.basicConfig(level=logging.INFO)
    
    manager = AlertManager()
    
    # Create test alert
    test_prediction = {
        'ticker': 'GME',
        'risk_level': 'high',
        'risk_score': 75,
        'manipulation_probability': 0.75,
        'volume_anomaly_prob': 0.8,
        'sentiment_surge_prob': 0.9,
        'lead_lag_score': 0.5,
        'model_version': 'test'
    }
    
    print("\nCreating test alert...")
    alert = manager.create_alert('GME', test_prediction)
    
    if alert:
        print(f"\nAlert created:")
        print(f"  Severity: {alert['severity']}")
        print(f"  Message: {alert['message']}")
        print(f"  Risk Score: {alert['risk_score']}")
    
    # Get alert stats
    print("\nAlert statistics (last 24 hours):")
    stats = manager.get_alert_stats(hours=24)
    print(f"  Total alerts: {stats['total_alerts']}")
    print(f"  By severity: {stats['by_severity']}")
    print(f"  Unacknowledged: {stats['unacknowledged_count']}")
    
    manager.close()
