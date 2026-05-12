"""
Tests for the stock-manipulation-detector project.
All tests use in-memory synthetic data — no network or file I/O required.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(n=60, seed=42) -> pd.DataFrame:
    """Generate a small synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC'),
        'Open':   close + rng.normal(0, 0.1, n),
        'High':   close + np.abs(rng.normal(0, 0.3, n)),
        'Low':    close - np.abs(rng.normal(0, 0.3, n)),
        'Close':  close,
        'Volume': rng.integers(1_000, 100_000, n).astype(float),
    })


# ── Test 1: MarketFeatureExtractor ────────────────────────────────────────────

def test_market_feature_extractor_runs():
    """MarketFeatureExtractor must return a DataFrame with expected columns."""
    from src.preprocessing.feature_extractor import MarketFeatureExtractor
    df = _make_ohlcv(60)
    extractor = MarketFeatureExtractor()
    result = extractor.calculate_technical_indicators(df)

    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert len(result) == 60, "Row count should be preserved"
    for col in ['RSI', 'MACD']:
        assert col in result.columns, f"Expected column '{col}' in result"


# ── Test 2: ManipulationLabeler ───────────────────────────────────────────────

def test_manipulation_labeler_output_columns():
    """ManipulationLabeler must produce is_manipulation and manipulation_confidence columns."""
    from src.preprocessing.feature_extractor import MarketFeatureExtractor
    from src.preprocessing.labeler import ManipulationLabeler

    df = _make_ohlcv(80)
    extractor = MarketFeatureExtractor()
    df = extractor.calculate_technical_indicators(df)

    labeler = ManipulationLabeler(
        price_spike_threshold=0.10,
        volume_threshold=2.0,
        crash_threshold=-0.08,
        time_window_hours=48,
    )
    labels = labeler.detect_pump_and_dump_pattern(df)

    assert isinstance(labels, pd.DataFrame), "Labels should be a DataFrame"
    assert 'is_manipulation' in labels.columns, "Expected 'is_manipulation' column"
    assert 'manipulation_confidence' in labels.columns, "Expected 'manipulation_confidence' column"
    assert labels['is_manipulation'].isin([0, 1]).all(), "Labels must be binary (0 or 1)"
    assert labels['manipulation_confidence'].between(0, 1).all(), \
        "Confidence scores must be in [0, 1]"


# ── Test 3: data_fusion_service feature builder ───────────────────────────────

def test_build_features_from_ohlcv():
    """_build_features_from_ohlcv must produce technical columns with no NaN/Inf."""
    from src.inference.data_fusion_service import _build_features_from_ohlcv

    df = _make_ohlcv(60)
    result = _build_features_from_ohlcv(df)

    assert isinstance(result, pd.DataFrame)
    for col in ['rsi', 'macd', 'volume_zscore', 'hl_spread']:
        assert col in result.columns, f"Expected column '{col}'"

    assert not result.isin([np.inf, -np.inf]).any().any(), "No Inf values expected"
    assert not result.isnull().any().any(), "No NaN values expected after build"


# ── Test 4: TemporalFusionTransformer forward pass ────────────────────────────

def test_transformer_forward_pass_keys():
    """TFT forward pass must return the exact output keys the predictor relies on."""
    import torch
    from src.models.transformer import TemporalFusionTransformer

    model = TemporalFusionTransformer(
        social_features=6, market_features=13,
        d_model=32, nhead=4, num_layers=2, dropout=0.0
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 24, 6), torch.zeros(1, 24, 13))

    expected_keys = {'manipulation_probability', 'lead_lag_score',
                     'volume_anomaly', 'sentiment_surge'}
    assert expected_keys.issubset(set(out.keys())), \
        f"Missing keys: {expected_keys - set(out.keys())}"

    prob = float(out['manipulation_probability'].squeeze())
    assert 0.0 <= prob <= 1.0, "manipulation_probability must be in [0, 1]"
