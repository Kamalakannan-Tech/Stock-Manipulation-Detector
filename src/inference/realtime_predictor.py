"""
Real-time prediction service using the trained model.
"""
import torch
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Optional
import os
import sys
from pathlib import Path

# Add project root to path (insert at front so src.* is always found first)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.transformer import TemporalFusionTransformer
from src.inference.data_fusion_service import DataFusionService
from src.utils.cache_fallback import get_redis_client
import json

# pymongo is optional — the predictor gracefully skips persistence if absent
try:
    from pymongo import MongoClient
    _PYMONGO_AVAILABLE = True
except ImportError:
    MongoClient = None          # type: ignore
    _PYMONGO_AVAILABLE = False

logger = logging.getLogger(__name__)

class RealtimePredictor:
    """
    Real-time manipulation detection using trained model.
    """
    
    def __init__(self,
                 model_path: str = 'models/saved_models/best_model.pth',
                 device: str = 'cpu',
                 mongo_uri: str = 'mongodb://localhost:27017/',
                 mongo_db: str = 'stock_manipulation',
                 redis_url: str = 'redis://localhost:6379'):
        """
        Initialize realtime predictor.
        
        Args:
            model_path: Path to trained model
            device: Device to run model on ('cpu' or 'cuda')
            mongo_uri: MongoDB connection URI
            mongo_db: MongoDB database name
            redis_url: Redis connection URL
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Initialize data fusion service
        self.fusion = DataFusionService(mongo_uri, mongo_db, redis_url)

        # MongoDB (optional - graceful fallback)
        self.predictions_collection = None
        try:
            self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
            self.mongo_client.admin.command('ping')
            self.db = self.mongo_client[mongo_db]
            self.predictions_collection = self.db['predictions']
            logger.info("MongoDB connected for predictions")
        except Exception as e:
            logger.warning(f"MongoDB unavailable ({e}), predictions won't be persisted")
            self.mongo_client = None
            self.db = None

        # Redis/cache with fallback
        self.redis_client = get_redis_client(redis_url, decode_responses=True)
        
        # Load model
        self.model = None
        self.model_loaded = False
        
        if os.path.exists(model_path):
            try:
                self.load_model(model_path)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        else:
            logger.warning(f"Model file not found: {model_path}")
            logger.warning("Predictor will operate in data-only mode")
    
    def load_model(self, model_path: str):
        """Load the trained model — reads architecture from checkpoint."""
        try:
            logger.info(f"Loading model from {model_path}...")

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Read architecture from checkpoint (new format) or default to compact model
            n_soc = checkpoint.get('social_features', 6)
            n_mkt = checkpoint.get('market_features', 13)
            d_mod = checkpoint.get('d_model', 64)
            nhead = checkpoint.get('nhead',   4)
            nlyr  = checkpoint.get('num_layers', 2)
            drp   = checkpoint.get('dropout',  0.1)
            self.best_threshold = float(checkpoint.get('best_threshold', 0.5))

            self.model = TemporalFusionTransformer(
                social_features=n_soc,
                market_features=n_mkt,
                d_model=d_mod,
                nhead=nhead,
                num_layers=nlyr,
                dropout=drp,
            )

            # Strip torch.compile(aot_eager) prefix if present
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if any(k.startswith('_orig_mod.') for k in state_dict):
                state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
                logger.info('Stripped _orig_mod. prefix from compiled checkpoint')
            self.model.load_state_dict(state_dict, strict=False)

            self.model.to(self.device)
            self.model.eval()

            self.model_loaded = True
            logger.info(
                f"Model loaded successfully (d_model={d_mod}, layers={nlyr}, "
                f"threshold={self.best_threshold:.2f})"
            )

        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            self.model_loaded = False


    def mc_dropout_predict(self, social_tensor: 'torch.Tensor',
                           market_tensor: 'torch.Tensor',
                           n_passes: int = 20) -> dict:
        """
        Monte Carlo Dropout inference.

        Runs the model N times with dropout ENABLED (stochastic mode) and
        aggregates the distribution of manipulation probabilities.

        Returns:
            {
              'mean'       : float  — expected manipulation probability
              'std'        : float  — epistemic uncertainty (lower = more certain)
              'ci_low'     : float  — 5th-percentile of the distribution
              'ci_high'    : float  — 95th-percentile
              'confidence' : float  — 1 - 2*std  (higher = more certain, capped [0,1])
            }
        """
        self.model.train()          # activates dropout
        samples = []
        with torch.no_grad():
            for _ in range(n_passes):
                out  = self.model(social_tensor, market_tensor)
                prob = float(out['manipulation_probability'].cpu().item())
                samples.append(prob)
        self.model.eval()           # restore eval mode

        arr  = np.array(samples, dtype=np.float32)
        mean = float(arr.mean())
        std  = float(arr.std())
        return {
            'mean':       mean,
            'std':        std,
            'ci_low':     float(np.percentile(arr, 5)),
            'ci_high':    float(np.percentile(arr, 95)),
            'confidence': float(max(0.0, min(1.0, 1.0 - 2 * std))),
        }

    
    def predict(self, ticker: str, use_mc_dropout: bool = True,
                mc_passes: int = 20) -> Optional[Dict]:
        """
        Make a prediction for a ticker.

        Args:
            ticker:         Stock ticker symbol
            use_mc_dropout: Run MC Dropout for uncertainty estimation (adds ~50 ms)
            mc_passes:      Number of stochastic forward passes for MC Dropout

        Returns:
            Dictionary with prediction results + confidence fields:
              manipulation_probability  float 0-1 (deterministic single pass)
              confidence_score          float 0-1 (MC mean; same if MC disabled)
              confidence_level          str   'Very High' | 'High' | 'Moderate' | 'Low' | 'Very Low'
              uncertainty               float 0-1 (MC std; 0 = perfectly certain)
              confidence_interval       [low, high] 90% credible interval from MC
        """
        try:
            # Get fused data
            model_input = self.fusion.prepare_model_input(ticker)

            if model_input is None:
                logger.warning(f"Cannot prepare input for {ticker}")
                return self._get_default_prediction(ticker, "insufficient_data")

            social_features, market_features = model_input

            # If model not loaded, return data-based analysis
            if not self.model_loaded:
                return self._analyze_without_model(ticker, social_features, market_features)

            # Prepare tensors
            social_tensor = torch.FloatTensor(social_features).unsqueeze(0).to(self.device)
            market_tensor = torch.FloatTensor(market_features).unsqueeze(0).to(self.device)

            # ── Deterministic single-pass prediction ───────────────────────────
            with torch.no_grad():
                outputs = self.model(social_tensor, market_tensor, return_attention=True)

            manipulation_prob  = float(outputs['manipulation_probability'].cpu().numpy()[0][0])
            lead_lag_score     = float(outputs['lead_lag_score'].cpu().numpy()[0][0])
            volume_anomaly     = float(outputs['volume_anomaly'].cpu().numpy()[0][0])
            sentiment_surge    = float(outputs['sentiment_surge'].cpu().numpy()[0][0])

            # ── MC Dropout confidence estimation ─────────────────────────────
            if use_mc_dropout and mc_passes > 1:
                mc = self.mc_dropout_predict(social_tensor, market_tensor, mc_passes)
                confidence_score    = mc['mean']
                uncertainty         = round(mc['std'],  4)
                confidence_interval = [round(mc['ci_low'], 4), round(mc['ci_high'], 4)]
                _conf_raw           = mc['confidence']
            else:
                confidence_score    = manipulation_prob
                uncertainty         = 0.0
                confidence_interval = [manipulation_prob, manipulation_prob]
                _conf_raw           = 1.0

            # ── Confidence level label ──────────────────────────────────────
            # Combines the probability magnitude AND how certain the model is.
            # High prob + low uncertainty = Very High confidence
            # High prob + high uncertainty = Moderate (model is unsure)
            if _conf_raw >= 0.85:
                confidence_level = 'Very High'
            elif _conf_raw >= 0.70:
                confidence_level = 'High'
            elif _conf_raw >= 0.50:
                confidence_level = 'Moderate'
            elif _conf_raw >= 0.30:
                confidence_level = 'Low'
            else:
                confidence_level = 'Very Low'

            # ── Risk score & level (based on MC mean probability) ───────────
            risk_score = confidence_score * 100
            if risk_score >= 75:
                risk_level = 'critical'
            elif risk_score >= 50:
                risk_level = 'high'
            elif risk_score >= 25:
                risk_level = 'medium'
            else:
                risk_level = 'low'

            prediction = {
                'ticker':                   ticker,
                'timestamp':                datetime.now(),
                # Core detection
                'manipulation_probability': round(manipulation_prob,  4),
                'risk_score':               round(risk_score,         2),
                'risk_level':               risk_level,
                # Confidence (new)
                'confidence_score':         round(confidence_score,   4),
                'confidence_level':         confidence_level,
                'uncertainty':              uncertainty,
                'confidence_interval':      confidence_interval,
                # Auxiliary signals
                'lead_lag_score':           round(lead_lag_score,     4),
                'volume_anomaly_prob':      round(volume_anomaly,     4),
                'sentiment_surge_prob':     round(sentiment_surge,    4),
                'model_version':            'v2.0',
                'data_quality':             'good',
            }
            
            # Store prediction in MongoDB (if available)
            if self.predictions_collection is not None:
                try:
                    self.predictions_collection.insert_one(prediction.copy())
                except Exception:
                    pass

            # Cache in Redis/memory
            redis_key = f"prediction:{ticker}"
            self.redis_client.setex(
                redis_key,
                300,  # 5 minute expiry
                json.dumps(prediction, default=str)
            )
            self.redis_client.publish(
                f"predictions:{ticker}",
                json.dumps(prediction, default=str)
            )
            
            logger.info(f"Prediction for {ticker}: risk={risk_level} ({risk_score:.1f}%), "
                       f"manipulation_prob={manipulation_prob:.3f}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}", exc_info=True)
            return self._get_default_prediction(ticker, "error")
    
    def _analyze_without_model(self, ticker: str, social_features: np.ndarray,
                               market_features: np.ndarray) -> Dict:
        """
        Rule-based analysis using market proxy features when model is not available.
        social_features[:, 0] = volume z-score
        social_features[:, 1] = price momentum 5-period
        social_features[:, 2] = RSI deviation from 0.5
        social_features[:, 3] = price acceleration
        social_features[:, 4] = HL spread
        social_features[:, 5] = volume ratio
        """
        # Use last time step of proxy features
        vol_zscore = float(np.mean(social_features[:, 0]))     # avg volume z-score
        price_mom = float(np.mean(social_features[:, 1]))      # avg price momentum
        rsi_dev = float(np.mean(social_features[:, 2]))        # avg RSI deviation
        acceleration = float(np.mean(social_features[:, 3]))   # avg price acceleration
        vol_ratio = float(np.mean(social_features[:, 5]))      # avg volume ratio

        risk_score = 0.0

        # Unusual volume (key pump-and-dump signal)
        if vol_zscore > 3.0:
            risk_score += 35
        elif vol_zscore > 2.0:
            risk_score += 20
        elif vol_zscore > 1.5:
            risk_score += 10

        # Strong positive price momentum (pump signal)
        if price_mom > 0.05:
            risk_score += 25
        elif price_mom > 0.02:
            risk_score += 12

        # Overbought RSI (RSI > 70 → rsi_dev > 0.2)
        if rsi_dev > 0.25:
            risk_score += 20
        elif rsi_dev > 0.15:
            risk_score += 10

        # Sudden price acceleration
        if abs(acceleration) > 0.05:
            risk_score += 15
        elif abs(acceleration) > 0.02:
            risk_score += 7

        # Volume ratio
        if vol_ratio > 0.8:  # > 4x average (ratio normalized to [0,1])
            risk_score += 5

        risk_score = min(risk_score, 100.0)

        if risk_score >= 75:
            risk_level = 'critical'
        elif risk_score >= 50:
            risk_level = 'high'
        elif risk_score >= 25:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        prediction = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'manipulation_probability': risk_score / 100,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'lead_lag_score': float(price_mom),
            'volume_anomaly_prob': min(vol_zscore / 4.0, 1.0),
            'sentiment_surge_prob': min(abs(rsi_dev) * 4.0, 1.0),
            'model_version': 'rule_based_v2',
            'data_quality': 'good'
        }

        logger.info(f"Rule-based analysis for {ticker}: risk={risk_level} ({risk_score:.1f}%), "
                    f"vol_z={vol_zscore:.2f}, mom={price_mom:.3f}")
        return prediction
    
    def _get_default_prediction(self, ticker: str, reason: str) -> Dict:
        """Return default prediction when analysis cannot be performed."""
        return {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'manipulation_probability': 0,
            'risk_score': 0,
            'risk_level': 'unknown',
            'lead_lag_score': 0,
            'volume_anomaly_prob': 0,
            'sentiment_surge_prob': 0,
            'model_version': 'none',
            'data_quality': reason
        }
    
    def get_latest_prediction(self, ticker: str) -> Optional[Dict]:
        """Get latest cached prediction for a ticker."""
        redis_key = f"prediction:{ticker}"
        data = self.redis_client.get(redis_key)
        if data:
            try:
                return json.loads(data)
            except Exception:
                pass

        # Fallback: check MongoDB
        if self.predictions_collection is not None:
            try:
                return self.predictions_collection.find_one(
                    {'ticker': ticker}, sort=[('timestamp', -1)]
                )
            except Exception:
                pass
        return None

    def get_prediction_history(self, ticker: str, limit: int = 50) -> list:
        """Get prediction history for a ticker."""
        if self.predictions_collection is None:
            return []
        try:
            return list(self.predictions_collection.find(
                {'ticker': ticker}
            ).sort('timestamp', -1).limit(limit))
        except Exception:
            return []
    
    def close(self):
        """Close connections."""
        self.fusion.close()
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            self.redis_client.close()


if __name__ == '__main__':
    # Test the predictor
    from dotenv import load_dotenv
    
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    
    predictor = RealtimePredictor(
        model_path=os.getenv('MODEL_PATH', 'models/saved_models/best_model.pth'),
        device=os.getenv('DEVICE', 'cpu')
    )
    
    # Test prediction
    ticker = 'GME'
    print(f"\nMaking prediction for ${ticker}...")
    
    prediction = predictor.predict(ticker)
    
    if prediction:
        print(f"\nPrediction Results:")
        print(f"  Risk Level: {prediction['risk_level'].upper()}")
        print(f"  Risk Score: {prediction['risk_score']:.1f}%")
        print(f"  Manipulation Probability: {prediction['manipulation_probability']:.3f}")
        print(f"  Model Version: {prediction['model_version']}")
        print(f"  Data Quality: {prediction['data_quality']}")
    else:
        print("Failed to make prediction")
    
    predictor.close()
