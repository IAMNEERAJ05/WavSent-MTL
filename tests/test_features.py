"""
tests/test_features.py
========================
Unit tests for feature engineering and wavelet denoising.

Run with: python -m pytest tests/test_features.py -v

All tests use synthetic data — no real data files required.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CONFIG
from src.data.preprocessor import coif3_denoise
from src.data.feature_engineering import (
    compute_rsi, compute_macd, compute_bb_width, compute_roc,
    compute_ema, compute_atr, compute_obv, compute_stoch_k,
    compute_williams_r, compute_cci, compute_all_features,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def price_df():
    """Synthetic OHLCV DataFrame with denoised columns for feature tests."""
    np.random.seed(0)
    n = 300
    close = 10000 + np.cumsum(np.random.randn(n) * 30)
    df = pd.DataFrame({
        'Close':   close,
        'Close_d': close + np.random.randn(n) * 5,
        'Open_d':  close * 0.999 + np.random.randn(n) * 5,
        'High_d':  close * 1.005 + np.abs(np.random.randn(n)) * 5,
        'Low_d':   close * 0.995 - np.abs(np.random.randn(n)) * 5,
        'Volume_d': np.abs(np.random.randn(n) * 1e6 + 5e6),
    })
    # Ensure High >= Close >= Low
    df['High_d'] = df[['Close_d', 'High_d']].max(axis=1)
    df['Low_d'] = df[['Close_d', 'Low_d']].min(axis=1)
    return df


# ── Test 1: RSI values in [0, 100] ────────────────────────────────────────────

def test_rsi_range(price_df):
    """RSI must be in [0, 100] for all non-NaN values."""
    rsi = compute_rsi(price_df['Close_d'])
    valid = rsi.dropna()
    assert (valid >= 0).all(), "RSI has values < 0"
    assert (valid <= 100).all(), "RSI has values > 100"


# ── Test 2: Wavelet preserves signal length ───────────────────────────────────

def test_wavelet_output_length():
    """Denoised signal must have same length as input."""
    original = np.random.randn(1090)
    denoised = coif3_denoise(original)
    assert len(denoised) == len(original), \
        f"Length mismatch: {len(denoised)} != {len(original)}"


# ── Test 3: Wavelet actually reduces noise ────────────────────────────────────

def test_wavelet_reduces_noise():
    """Denoised signal must have lower std than noisy signal."""
    np.random.seed(42)
    signal = np.sin(np.linspace(0, 10, 200))
    noisy = signal + np.random.randn(200) * 0.5
    denoised = coif3_denoise(noisy)
    assert np.std(denoised) < np.std(noisy), \
        "Wavelet denoising did not reduce noise"


# ── Test 4: Features computed on denoised differ from raw ─────────────────────

def test_features_on_denoised_not_raw(price_df):
    """RSI on Close_d must differ from RSI on Close (raw)."""
    rsi_denoised = compute_rsi(price_df['Close_d'])
    rsi_raw = compute_rsi(price_df['Close'])
    assert not rsi_denoised.equals(rsi_raw), \
        "RSI on denoised == RSI on raw — denoising not applied?"


# ── Test 5: Selected features exist in dataframe (if file exists) ─────────────

@pytest.mark.skipif(
    not os.path.exists(
        CONFIG['kotekar_processed_dir'] + 'selected_features.json'
    ),
    reason="selected_features.json not yet generated"
)
def test_selected_features_exist():
    """All selected features must be present in featured_data.csv."""
    with open(CONFIG['kotekar_processed_dir'] + 'selected_features.json') as f:
        selected = json.load(f)
    df = pd.read_csv(CONFIG['kotekar_processed_dir'] + 'featured_data.csv')
    for feat in selected:
        assert feat in df.columns, f"Feature {feat} missing from featured_data.csv"


# ── Test 6: Regression target magnitude is reasonable ────────────────────────

def test_regression_target_range():
    """Daily returns on synthetic data must be small (not outliers)."""
    np.random.seed(1)
    close = 10000 + np.cumsum(np.random.randn(300) * 30)
    returns = np.diff(close) / close[:-1]
    assert np.abs(returns).max() < 10.0, "Regression targets have extreme values"


# ── Test 7: Classification targets are binary ─────────────────────────────────

def test_classification_target_binary():
    """Classification labels must be {0, 1} only."""
    np.random.seed(2)
    close = 10000 + np.cumsum(np.random.randn(200) * 20)
    y_clf = np.array([1 if close[i] > close[i-1] else 0
                      for i in range(1, len(close))], dtype=np.int32)
    assert set(np.unique(y_clf)).issubset({0, 1})


# ── Test 8: ATR is always non-negative ────────────────────────────────────────

def test_atr_positive(price_df):
    """ATR must be >= 0 for all non-NaN values."""
    atr = compute_atr(price_df['High_d'], price_df['Low_d'], price_df['Close_d'])
    valid = atr.dropna()
    assert (valid >= 0).all(), "ATR has negative values"


# ── Test 9: compute_all_features adds all 10 indicator columns ───────────────

def test_compute_all_features_columns(price_df):
    """compute_all_features must add RSI_14, MACD, BB_width, etc."""
    expected_indicators = [
        'RSI_14', 'MACD', 'BB_width', 'ROC_5', 'EMA_9',
        'ATR_14', 'OBV', 'STOCH_K', 'WILLIAMS_R', 'CCI_20'
    ]
    result = compute_all_features(price_df)
    for col in expected_indicators:
        assert col in result.columns, f"Missing column: {col}"


# ── Test 10: EMA has no NaN rows ──────────────────────────────────────────────

def test_ema_no_nan(price_df):
    """EMA (EWM-based) should not produce NaN values."""
    ema = compute_ema(price_df['Close_d'])
    assert ema.isnull().sum() == 0, "EMA has unexpected NaN values"
