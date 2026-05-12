"""
Feature extraction for market and social data.
All 13 market + 6 social dimensions as per spec.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MarketFeatureExtractor:
    """Extract and compute 13-dimensional market feature vector."""

    MARKET_FEATURES = [
        'Close', 'Returns_1h', 'Returns_4h', 'Price_Momentum_5h',
        'SMA_5', 'SMA_20', 'EMA_12', 'EMA_26',
        'Volume', 'Volume_MA', 'Volume_Std', 'Abnormal_Volume_Zscore',
        'Volatility_5h',
    ]

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all market technical indicators in-place."""
        if df.empty or 'Close' not in df.columns:
            return df
        try:
            c = df['Close']

            # Returns
            df['Returns_1h'] = c.pct_change(1).fillna(0)
            df['Returns_4h'] = c.pct_change(4).fillna(0)

            # Moving averages
            df['SMA_5']  = c.rolling(5,  min_periods=1).mean()
            df['SMA_20'] = c.rolling(20, min_periods=1).mean()
            df['EMA_12'] = c.ewm(span=12, adjust=False).mean()
            df['EMA_26'] = c.ewm(span=26, adjust=False).mean()

            # Momentum
            df['Price_Momentum_5h']  = c.pct_change(5).fillna(0)
            df['Price_Momentum_20h'] = c.pct_change(20).fillna(0)

            # Volatility
            ret = c.pct_change()
            df['Volatility_5h']  = ret.rolling(5,  min_periods=1).std().fillna(0)
            df['Volatility_20h'] = ret.rolling(20, min_periods=1).std().fillna(0)

            # Volume features
            if 'Volume' in df.columns:
                v = df['Volume'].fillna(0)
                df['Volume_MA']  = v.rolling(20, min_periods=1).mean()
                df['Volume_Std'] = v.rolling(20, min_periods=1).std().fillna(0)
                df['Abnormal_Volume_Zscore'] = (
                    (v - df['Volume_MA']) / (df['Volume_Std'] + 1e-9)
                ).clip(-5, 5)

            # Bollinger Bands
            df['BB_Middle'] = c.rolling(20, min_periods=1).mean()
            bb_std = c.rolling(20, min_periods=1).std().fillna(0)
            df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
            df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Middle'] + 1e-9)

            # RSI
            delta = c.diff()
            gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
            loss = (-delta).clip(lower=0).rolling(14, min_periods=1).mean()
            rs = gain / (loss + 1e-9)
            df['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

            # High-Low spread
            if 'High' in df.columns and 'Low' in df.columns:
                df['HL_Spread'] = (df['High'] - df['Low']) / (c + 1e-9)
                df['Price_Position'] = (c - df['Low']) / (
                    (df['High'] - df['Low']).replace(0, 1e-9)
                )

            # Price change percentage
            df['Price_Change_Pct'] = c.pct_change().fillna(0)

            df = df.ffill().bfill().fillna(0)
            return df
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return df

    def build_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Return (N, 13) numpy array of normalised market features."""
        cols = [c for c in self.MARKET_FEATURES if c in df.columns]
        mat = df[cols].fillna(0).replace([np.inf, -np.inf], 0).values

        # Pad to exactly 13 cols
        if mat.shape[1] < 13:
            mat = np.hstack([mat, np.zeros((len(mat), 13 - mat.shape[1]))])
        return mat[:, :13].astype(np.float32)


class SocialFeatureExtractor:
    """
    Build 6-dimensional social feature vector.
    Uses real social data if available, otherwise derives proxies from market
    data (volume + volatility + RSI) so the pipeline never breaks.
    """

    SOCIAL_COLS = [
        'sentiment_score', 'post_volume', 'engagement_score',
        'bot_activity_score', 'narrative_similarity', 'sentiment_velocity',
    ]

    def build_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Return (N, 6) numpy array of social/proxy features."""
        if all(c in df.columns for c in self.SOCIAL_COLS[:3]):
            return self._from_explicit(df)
        return self._from_market_proxy(df)

    def _from_explicit(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c if c in df.columns else None for c in self.SOCIAL_COLS]
        parts = []
        for col in self.SOCIAL_COLS:
            if col in df.columns:
                v = df[col].fillna(0).values.astype(np.float64)
            else:
                v = np.zeros(len(df))
            parts.append(v)
        mat = np.column_stack(parts).astype(np.float32)
        return self._normalize(mat)

    def _from_market_proxy(self, df: pd.DataFrame) -> np.ndarray:
        """Derive 6 pseudo-social features from pure market signals."""
        n = len(df)
        close  = df['Close'].ffill().fillna(0).values if 'Close' in df.columns else np.ones(n)
        volume = df['Volume'].fillna(0).values        if 'Volume' in df.columns else np.zeros(n)

        # 1. Volume z-score → proxy for post volume surge
        vm = pd.Series(volume).rolling(20, min_periods=1).mean().values
        vs = pd.Series(volume).rolling(20, min_periods=1).std().fillna(1).values
        vol_z = np.clip((volume - vm) / (vs + 1e-9), -5, 5)

        # 2. Price momentum 5h → proxy for sentiment
        mom = np.zeros(n)
        mom[5:] = (close[5:] - close[:-5]) / (np.abs(close[:-5]) + 1e-9)
        mom = np.clip(mom, -1, 1)

        # 3. RSI deviation (RSI/100 - 0.5) → proxy for sentiment strength
        delta = np.diff(close, prepend=close[0])
        gain = pd.Series(np.where(delta > 0, delta, 0)).ewm(span=14).mean().values
        loss = pd.Series(np.where(delta < 0, -delta, 0)).ewm(span=14).mean().values
        rsi = 100 - (100 / (1 + gain / (loss + 1e-9)))
        rsi_dev = np.clip(rsi / 100.0 - 0.5, -0.5, 0.5)

        # 4. Price acceleration → proxy for bot-driven spikes
        ret = np.zeros(n); ret[1:] = np.diff(close) / (np.abs(close[:-1]) + 1e-9)
        acc = np.zeros(n); acc[1:] = np.diff(ret)
        acc = np.clip(acc, -0.1, 0.1) * 10

        # 5. High-Low spread / close → proxy for copypasta coordination noise
        high = df['High'].fillna(0).values if 'High' in df.columns else close
        low  = df['Low'].fillna(0).values  if 'Low'  in df.columns else close
        hl = np.clip((high - low) / (close + 1e-9), 0, 0.5)

        # 6. Volume ratio vs rolling mean → proxy for engagement
        vol_ratio = np.clip(volume / (vm + 1e-9), 0, 10) / 10

        mat = np.column_stack([vol_z, mom, rsi_dev, acc, hl, vol_ratio]).astype(np.float32)
        return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        try:
            return scaler.fit_transform(mat).astype(np.float32)
        except Exception:
            return mat


class FinBERTSentimentAnalyzer:
    """Load ProsusAI/finbert for real sentiment analysis (optional)."""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            # Auto-select GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.model = self.model.to(self.device)  # Move model to GPU
            self.model.eval()
            self._torch = torch
            logger.info(f"FinBERT loaded successfully on {self.device}")
        except Exception as e:
            logger.warning(f"FinBERT not available (will use proxy features): {e}")

    @property
    def available(self) -> bool:
        return self.model is not None

    def analyze_sentiment(self, text: str) -> dict:
        if not self.available or not text:
            return {'sentiment_score': 0.0, 'confidence': 0.0, 'label': 'neutral'}
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                    max_length=512, padding=True)
            # Move input tensors to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                logits = self.model(**inputs).logits
            probs = self._torch.nn.functional.softmax(logits, dim=-1)[0]
            # FinBERT labels: positive=0, negative=1, neutral=2
            score = probs[0].item() - probs[1].item()
            labels = ['positive', 'negative', 'neutral']
            label = labels[probs.argmax().item()]
            return {'sentiment_score': score, 'confidence': probs.max().item(), 'label': label}
        except Exception as e:
            logger.debug(f"Sentiment analysis error: {e}")
            return {'sentiment_score': 0.0, 'confidence': 0.0, 'label': 'neutral'}

    def batch_analyze(self, texts: list, batch_size: int = 16) -> list:
        """Batch inference for speed — single GPU forward pass per batch."""
        if not self.available or not texts:
            return [{'sentiment_score': 0.0, 'confidence': 0.0, 'label': 'neutral'} for _ in texts]
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                inputs = self.tokenizer(
                    batch, return_tensors="pt", truncation=True,
                    max_length=512, padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with self._torch.no_grad():
                    logits = self.model(**inputs).logits
                probs = self._torch.nn.functional.softmax(logits, dim=-1)  # [B, 3]
                labels_list = ['positive', 'negative', 'neutral']
                for p in probs:
                    score = p[0].item() - p[1].item()
                    label = labels_list[p.argmax().item()]
                    results.append({'sentiment_score': score, 'confidence': p.max().item(), 'label': label})
            except Exception as e:
                logger.debug(f"Batch sentiment error: {e}")
                results.extend([{'sentiment_score': 0.0, 'confidence': 0.0, 'label': 'neutral'} for _ in batch])
        return results
