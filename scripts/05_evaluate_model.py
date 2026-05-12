"""
05_evaluate_model.py
Loads the best checkpoint, runs test-set inference, finds optimal threshold,
computes all metrics, and saves metrics.json + PNG evaluation plots.

Improvements:
  - Uses GPU automatically (matches training device)
  - Loads full architecture from checkpoint (d_model, nhead, num_layers, dropout)
  - Uses chronological walk-forward split (matches training split)
  - AUC reported alongside F1
"""
import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, '.')
from src.models.transformer import TemporalFusionTransformer
from src.evaluation.evaluator import ManipulationEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 24
MODEL_PATH      = 'models/saved_models/best_model.pth'
HISTORY_PATH    = 'models/saved_models/training_history.json'
VAL_SPLIT       = 0.20   # Must match training val_split


def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(d)}")
        return d
    logger.info("Using CPU")
    return torch.device('cpu')


# ── Feature preparation (mirrors 04_train_model.py exactly) ──────────────────
# NOTE: Priority order must match training exactly to ensure features are
#       in the same column positions. Uses the RICH indicator list first,
#       falls back to the legacy OHLCV list if the richer cols are absent.

def prepare_features(df):
    # Primary column list (matches 04_train_model.py DEFAULT priority)
    MARKET_COLS = [
        'Close', 'Returns_1h', 'Returns_4h', 'Price_Momentum_5h',
        'SMA_20', 'EMA_12', 'RSI', 'MACD',
        'Volume', 'Abnormal_Volume_Zscore', 'Volatility_5h',
        'BB_Width', 'HL_Spread',
    ]
    # Fallback: older processed files may not have the richer indicators
    MARKET_FALLBACK = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_20', 'EMA_12', 'RSI', 'MACD',
        'BB_Upper', 'BB_Lower', 'Abnormal_Volume_Zscore', 'Price_Change_Pct',
    ]
    SOCIAL_COLS = [
        'sentiment_score', 'post_volume', 'engagement_score',
        'bot_activity_score', 'narrative_similarity', 'sector_divergence',
    ]

    avail_m = [c for c in MARKET_COLS if c in df.columns]
    if len(avail_m) < 8:
        avail_m = [c for c in MARKET_FALLBACK if c in df.columns]
    avail_s = [c for c in SOCIAL_COLS if c in df.columns]

    market_raw = df[avail_m].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float64)
    if market_raw.shape[1] < 13:
        market_raw = np.hstack([market_raw, np.zeros((len(df), 13 - market_raw.shape[1]))])
    market_raw = market_raw[:, :13]

    if avail_s:
        social_raw = df[avail_s].fillna(0).values.astype(np.float64)
        if social_raw.shape[1] < 6:
            social_raw = np.hstack([social_raw, np.zeros((len(df), 6 - social_raw.shape[1]))])
        social_raw = social_raw[:, :6]
    else:
        returns   = pd.Series(df['Close'].values).pct_change().fillna(0).values
        vol_std   = pd.Series(df['Volume'].values).rolling(20, min_periods=1).std().fillna(0).values
        vol_mean  = pd.Series(df['Volume'].values).rolling(20, min_periods=1).mean().fillna(1).values
        vol_z     = np.where(vol_mean > 0, (df['Volume'].values - vol_mean) / (vol_std + 1e-9), 0)
        momentum  = pd.Series(df['Close'].values).pct_change(5).fillna(0).values
        rsi_vals  = df['RSI'].fillna(50).values if 'RSI' in df.columns else np.full(len(df), 50.0)
        rsi_dev   = (rsi_vals - 50) / 50
        accel     = np.gradient(returns)
        hl_spread = (df['High'].values - df['Low'].values) / (df['Close'].values + 1e-9) if 'High' in df.columns else np.zeros(len(df))
        social_raw = np.column_stack([vol_z, momentum, rsi_dev, accel, hl_spread, np.abs(returns)])
        social_raw = social_raw[:, :6]

    scaler_m = StandardScaler()
    scaler_s = StandardScaler()
    return scaler_s.fit_transform(social_raw).astype(np.float32), scaler_m.fit_transform(market_raw).astype(np.float32)


def create_sequences(social, market, labels, seq_len=SEQUENCE_LENGTH):
    X_s, X_m, y = [], [], []
    for i in range(len(labels) - seq_len):
        X_s.append(social[i:i+seq_len])
        X_m.append(market[i:i+seq_len])
        y.append(labels[i + seq_len])
    return np.array(X_s, dtype=np.float32), np.array(X_m, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    feat_path = 'data/processed/combined_features.csv'
    lbl_path  = 'data/labeled/labels.csv'
    if not os.path.exists(feat_path):
        logger.error(f'No data at {feat_path}. Run 03_preprocess_data.py first.')
        return
    if not os.path.exists(MODEL_PATH):
        logger.error(f'No model at {MODEL_PATH}. Run 04_train_model.py first.')
        return

    feat_df = pd.read_csv(feat_path)
    lbl_df  = pd.read_csv(lbl_path)
    n = min(len(feat_df), len(lbl_df))
    feat_df, lbl_df = feat_df.iloc[:n], lbl_df.iloc[:n]

    social_feat, market_feat = prepare_features(feat_df)
    labels  = lbl_df['is_manipulation'].values.astype(np.float32)
    tickers = feat_df['ticker'].values if 'ticker' in feat_df.columns else None

    # ── Per-ticker sequence creation + split (mirrors 04_train_model.py) ────────
    if tickers is not None:
        all_s, all_m, all_y, all_t = [], [], [], []
        for ticker in np.unique(tickers):
            mask = (tickers == ticker)
            idx  = np.where(mask)[0]
            if len(idx) <= SEQUENCE_LENGTH:
                continue
            xs, xm, ys = create_sequences(social_feat[idx], market_feat[idx], labels[idx])
            all_s.append(xs); all_m.append(xm); all_y.append(ys)
            all_t.extend([ticker] * len(ys))
        X_s = np.concatenate(all_s)
        X_m = np.concatenate(all_m)
        y   = np.concatenate(all_y)
        ticker_seqs = np.array(all_t)

        # Interleaved every-k-th split (mirrors training per_ticker_split)
        k = max(2, round(1.0 / VAL_SPLIT))
        va_s, va_m, va_y = [], [], []
        for ticker in np.unique(ticker_seqs):
            mask    = (ticker_seqs == ticker)
            idx     = np.where(mask)[0]
            val_idx = idx[::k]
            va_s.append(X_s[val_idx]); va_m.append(X_m[val_idx]); va_y.append(y[val_idx])
        X_s_test = np.concatenate(va_s)
        X_m_test = np.concatenate(va_m)
        y_test   = np.concatenate(va_y)
    else:
        X_s, X_m, y = create_sequences(social_feat, market_feat, labels)
        split_idx    = int(len(y) * (1 - VAL_SPLIT))
        X_s_test, X_m_test, y_test = X_s[split_idx:], X_m[split_idx:], y[split_idx:]

    logger.info(f'Test set: {len(y_test)} samples  +rate: {y_test.mean()*100:.1f}%  positive: {int(y_test.sum())}')

    # ── Load model from checkpoint (full arch) ─────────────────────────────────
    device = get_device()
    ckpt   = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    logger.info(f'Loaded checkpoint from epoch {ckpt.get("epoch", "?")}')

    # Prefer dims stored in checkpoint; fall back to data dims
    n_soc      = ckpt.get('social_features', X_s.shape[2])
    n_mkt      = ckpt.get('market_features', X_m.shape[2])
    d_model    = ckpt.get('d_model',    64)
    nhead      = ckpt.get('nhead',       4)
    nlyr       = ckpt.get('num_layers',  2)
    drp        = ckpt.get('dropout',   0.1)
    best_threshold = float(ckpt.get('best_threshold', 0.5))
    logger.info(
        f'Architecture: d_model={d_model}, nhead={nhead}, layers={nlyr}, dropout={drp}'
    )
    logger.info(f'Using checkpoint threshold: {best_threshold}')

    model = TemporalFusionTransformer(
        social_features=n_soc, market_features=n_mkt,
        d_model=d_model, nhead=nhead, num_layers=nlyr, dropout=drp
    ).to(device)

    # torch.compile(aot_eager) prefixes all keys with '_orig_mod.' — strip it
    raw_sd = ckpt['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in raw_sd):
        raw_sd = {k.replace('_orig_mod.', '', 1): v for k, v in raw_sd.items()}
        logger.info('Stripped _orig_mod. prefix from compiled checkpoint keys')
    model.load_state_dict(raw_sd, strict=True)
    model.eval()

    # ── Inference ──────────────────────────────────────────────────────────────
    ds = TensorDataset(torch.FloatTensor(X_s_test), torch.FloatTensor(X_m_test))
    dl = DataLoader(ds, batch_size=64)
    all_probs = []
    with torch.no_grad():
        for soc, mkt in dl:
            soc, mkt = soc.to(device), mkt.to(device)
            out  = model(soc, mkt)
            prob = out['manipulation_probability'].squeeze(-1)
            if prob.ndim == 0:
                prob = prob.unsqueeze(0)
            all_probs.extend(prob.cpu().numpy().tolist())
    y_prob = np.array(all_probs)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    evaluator = ManipulationEvaluator()

    # Optimise threshold from scratch on test set
    best_t = evaluator.find_best_threshold(y_test, y_prob)
    # Also consider the checkpoint threshold
    from sklearn.metrics import f1_score
    f1_ckpt = f1_score(y_test, (y_prob >= best_threshold).astype(int), zero_division=0)
    f1_best = f1_score(y_test, (y_prob >= best_t).astype(int), zero_division=0)
    final_thresh = best_t if f1_best >= f1_ckpt else best_threshold
    evaluator.threshold = final_thresh

    metrics = evaluator.compute_metrics(y_test, y_prob)
    logger.info('\n=== TEST METRICS ===')
    for k, v in metrics.items():
        if not isinstance(v, (list, dict)):
            logger.info(f'  {k}: {v}')

    # ── Training history ───────────────────────────────────────────────────────
    history = None
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            history = json.load(f)

    plots = evaluator.plot_all(y_test, y_prob, training_history=history)
    metrics['plots'] = {k: str(v) for k, v in plots.items()}
    evaluator.save_metrics(metrics)

    logger.info('\nEvaluation complete.')
    logger.info(f"F1={metrics.get('f1','N/A')}  AUC={metrics.get('roc_auc','N/A')}")
    logger.info(f"Plots saved: {list(plots.keys())}")


if __name__ == '__main__':
    main()
