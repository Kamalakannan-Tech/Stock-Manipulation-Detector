import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ManipulationLabeler:
    def __init__(self, price_spike_threshold=0.15,  # 15% price increase (realistic for pump)
                 volume_threshold=3.5,              # 3.5x normal volume (stricter spike)
                 crash_threshold=-0.10,             # 10% drop after spike (dump)
                 time_window_hours=48):             # 48 hours window
        self.price_spike_threshold = price_spike_threshold
        self.volume_threshold      = volume_threshold
        self.crash_threshold       = crash_threshold
        self.time_window           = time_window_hours

    def detect_pump_and_dump_pattern(self, df):
        if df.empty or 'Close' not in df.columns:
            logger.warning("Empty dataframe")
            return pd.DataFrame({
                'is_manipulation': [0] * max(1, len(df)),
                'has_volume_anomaly': [0] * max(1, len(df)),
                'has_sentiment_surge': [0] * max(1, len(df)),
                'manipulation_confidence': [0.0] * max(1, len(df))
            })

        labels = []
        for i in range(len(df)):
            label_info = {
                'is_manipulation': 0,
                'has_volume_anomaly': 0,
                'has_sentiment_surge': 0,
                'manipulation_confidence': 0.0
            }

            # Past-looking signals (last 12 hours/candles)
            recent_start = max(0, i - 12)
            recent_slice = df.iloc[recent_start:i]
            if len(recent_slice) >= 6:  # Require at least some recent data
                recent_prices = recent_slice['Close']
                recent_return = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / (recent_prices.iloc[0] + 1e-10)
                recent_volumes = recent_slice['Abnormal_Volume_Zscore'] if 'Abnormal_Volume_Zscore' in recent_slice.columns else pd.Series([0] * len(recent_slice))
                recent_vol_spike = recent_volumes.max() > self.volume_threshold * 0.7  # Slightly lenient for recent
                
                # Check for social sentiment surge BEFORE the price spike
                recent_sentiment = recent_slice['sentiment_score'] if 'sentiment_score' in recent_slice.columns else pd.Series([0])
                recent_posts = recent_slice['post_volume'] if 'post_volume' in recent_slice.columns else pd.Series([0])
                has_social_surge = (recent_sentiment.max() > 0.3) or (recent_posts.max() > 2.0)
            else:
                recent_return = 0
                recent_vol_spike = False
                has_social_surge = False

            # Future window - need enough data to detect pattern
            future_window = min(self.time_window, len(df) - i - 1)
            if future_window < 24:  # Need at least 24 hours of future data
                labels.append(label_info)
                continue

            current_price = df.iloc[i]['Close']
            future_slice = df.iloc[i:i + future_window]
            future_prices = future_slice['Close']

            # Volume check
            future_volumes = future_slice['Abnormal_Volume_Zscore'] if 'Abnormal_Volume_Zscore' in future_slice.columns else pd.Series([0] * len(future_slice))

            # Price spike detection
            max_future_price = future_prices.max()
            price_increase = (max_future_price - current_price) / (current_price + 1e-10)

            # Volume spike
            max_volume_zscore = future_volumes.max()
            has_volume_spike = max_volume_zscore > self.volume_threshold

            # Crash detection (optional)
            has_crash = False
            if price_increase > self.price_spike_threshold:
                spike_idx = future_prices.idxmax()
                spike_location = future_prices.index.get_loc(spike_idx)
                if spike_location < len(future_prices) - 5:
                    post_spike_prices = future_prices.iloc[spike_location:]
                    min_post_spike = post_spike_prices.min()
                    crash_magnitude = (min_post_spike - max_future_price) / (max_future_price + 1e-10)
                    has_crash = crash_magnitude < self.crash_threshold

            # Stricter classification - require strong signals
            confidence_score = 0.0

            # Price movement (primary signal)
            if price_increase > self.price_spike_threshold:
                confidence_score += 0.4  # Strong price spike
            elif price_increase > self.price_spike_threshold * 0.7:
                confidence_score += 0.2  # Moderate price increase

            # Volume spike (important signal)
            if has_volume_spike:
                confidence_score += 0.3  # Strong volume spike
            elif max_volume_zscore > self.volume_threshold * 0.7:
                confidence_score += 0.1  # Moderate volume

            # Crash after spike (confirms pump-and-dump)
            if has_crash:
                confidence_score += 0.3  # Strong indicator

            # Recent signals (supporting evidence)
            if recent_vol_spike:
                confidence_score += 0.1
            if recent_return > 0.08:  # 8% recent increase
                confidence_score += 0.1

            # STRICTURE: Social priming is supporting evidence, not a trump card.
            # A real pump-and-dump needs BOTH price action AND social pumping.
            # Add a moderate bonus for confirmed social signal; penalise its absence.
            if has_social_surge:
                confidence_score += 0.2   # Supporting evidence (was 0.4 — too generous)
            else:
                # No observed social pumping — likely organic volatility (earnings, news)
                confidence_score -= 0.3   # Moderate penalty (was -0.5 — too harsh for proxy social)

            # Require strong confidence for manipulation label.
            # Threshold = 0.75 means we need BOTH a strong price spike (0.4)
            # AND a volume spike (0.3) AND a crash (0.3) – or a similar 3-signal combination.
            if confidence_score >= 0.75:  # Stricter threshold to avoid over-labeling
                label_info['is_manipulation']      = 1
                label_info['manipulation_confidence'] = min(confidence_score, 1.0)
                label_info['has_volume_anomaly']   = int(has_volume_spike or recent_vol_spike)
                label_info['has_sentiment_surge']  = int(has_social_surge)

            labels.append(label_info)

        labels_df = pd.DataFrame(labels)
        n_manipulation = labels_df['is_manipulation'].sum()
        pos_rate = n_manipulation / max(len(labels_df), 1)
        logger.info(
            f"Labeled {len(labels_df)} samples: {n_manipulation} manipulation "
            f"({pos_rate * 100:.1f}%) | neg={labels_df['is_manipulation'].eq(0).sum()}"
        )
        return labels_df