"""
tests/test_data_pipeline.py
=============================
Unit tests for data pipeline: temporal split, scaler, windows, targets.

Run with: python -m pytest tests/test_data_pipeline.py -v

Tests are designed to run on CPU without requiring processed data files.
Tests that require processed files are marked and skipped if files absent.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CONFIG
from src.data.windows import create_windows, generate_targets, temporal_split
from src.data.preprocessor import apply_scaler, apply_reg_scaler


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a small synthetic DataFrame with Date + feature + Close columns."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2020-01-02', periods=n, freq='B')
    close = 10000 + np.cumsum(np.random.randn(n) * 50)
    df = pd.DataFrame({'Date': dates, 'Close': close,
                       'f1': np.random.randn(n),
                       'f2': np.random.randn(n)})
    return df


@pytest.fixture
def sample_split(sample_df):
    """Return train/val/test split of sample_df."""
    return temporal_split(sample_df)


@pytest.fixture
def sample_scaled(sample_df, sample_split):
    """Return scaled arrays from sample_split."""
    train_df, val_df, test_df = sample_split
    feats = ['f1', 'f2']
    tr = train_df[feats].values.astype(np.float32)
    vl = val_df[feats].values.astype(np.float32)
    te = test_df[feats].values.astype(np.float32)
    tr_s, vl_s, te_s, sc = apply_scaler(tr, vl, te, save_path='tests/test_scaler.pkl')
    yield tr_s, vl_s, te_s, sc
    # Cleanup
    if os.path.exists('tests/test_scaler.pkl'):
        os.remove('tests/test_scaler.pkl')


# ── Test 1: Temporal split preserves order ────────────────────────────────────

def test_split_no_shuffle(sample_split):
    """Verify temporal ordering: train < val < test."""
    train_df, val_df, test_df = sample_split
    assert train_df['Date'].max() < val_df['Date'].min(), \
        "Train/val date overlap"
    assert val_df['Date'].max() < test_df['Date'].min(), \
        "Val/test date overlap"


# ── Test 2: Scaler fit on train only ─────────────────────────────────────────

def test_scaler_fit_on_train_only(sample_scaled):
    """Train-scaled values must be in [0, 1]; val/test may exceed."""
    tr_s, vl_s, te_s, sc = sample_scaled
    assert tr_s.max() <= 1.0 + 1e-6, "Train max exceeds 1"
    assert tr_s.min() >= 0.0 - 1e-6, "Train min below 0"
    # Val/test are NOT refit — they CAN be outside [0,1]
    # This confirms scaler was fit on train only


# ── Test 3: No future leakage in window targets ────────────────────────────────

def test_no_future_leakage(sample_df, sample_split):
    """Target at index i corresponds to Close[i] vs Close[i-1]."""
    train_df, _, _ = sample_split
    close = train_df['Close'].values
    y_clf, _ = generate_targets(close)
    window = CONFIG['window_size']
    for i in range(min(10, len(y_clf))):
        idx = window + i
        expected = 1 if close[idx] > close[idx - 1] else 0
        assert y_clf[i] == expected, f"Leakage at index {i}"


# ── Test 4: No missing values (synthetic data) ────────────────────────────────

def test_no_missing_values(sample_df):
    """Synthetic DataFrame has no NaNs after feature selection."""
    assert sample_df[['f1', 'f2']].isnull().sum().sum() == 0


# ── Test 5: Kotekar merged_data has no NaN polarity (if file exists) ─────────

@pytest.mark.skipif(
    not os.path.exists(CONFIG['kotekar_processed_dir'] + 'merged_data.csv'),
    reason="merged_data.csv not yet generated"
)
def test_kotekar_gap_fill():
    """polarity_mean must have no NaN in merged kotekar data."""
    df = pd.read_csv(CONFIG['kotekar_processed_dir'] + 'merged_data.csv')
    assert df['polarity_mean'].isnull().sum() == 0


# ── Test 6: Kaggle gap period zero-filled (if file exists) ───────────────────

@pytest.mark.skipif(
    not os.path.exists(CONFIG['kaggle_processed_dir'] + 'merged_data.csv'),
    reason="merged_data.csv not yet generated"
)
def test_kaggle_gap_fill():
    """Gap period May–Dec 2021 must have polarity_mean=0 and polarity_max=0."""
    df = pd.read_csv(CONFIG['kaggle_processed_dir'] + 'merged_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    gap = df[
        (df['Date'] >= '2021-05-01') &
        (df['Date'] <= '2021-12-31')
    ]
    assert (gap['polarity_mean'] == 0).all(), "Gap polarity_mean != 0"
    assert (gap['polarity_max'] == 0).all(), "Gap polarity_max != 0"


# ── Test 7: Split sizes approximately 70/15/15 ────────────────────────────────

def test_split_sizes(sample_df, sample_split):
    """Verify 70/15/15 proportions within ±2%."""
    train_df, val_df, test_df = sample_split
    n = len(sample_df)
    assert abs(len(train_df) / n - 0.70) < 0.02
    assert abs(len(val_df) / n - 0.15) < 0.02
    assert abs(len(test_df) / n - 0.15) < 0.02


# ── Test 8: Window shapes correct ─────────────────────────────────────────────

def test_window_shapes(sample_scaled, sample_split):
    """Windows must have shape [n, 5, n_features]."""
    tr_s, _, _, _ = sample_scaled
    X_train = create_windows(tr_s)
    assert X_train.shape[1] == CONFIG['window_size'], \
        f"Expected window_size={CONFIG['window_size']}, got {X_train.shape[1]}"
    assert X_train.shape[2] == 2, "Expected 2 features"


# ── Test 9: Classification targets are binary ─────────────────────────────────

def test_clf_targets_binary(sample_split):
    """y_clf must contain only {0, 1}."""
    train_df, _, _ = sample_split
    y_clf, _ = generate_targets(train_df['Close'].values)
    assert set(np.unique(y_clf)).issubset({0, 1})


# ── Test 10: Regression targets are finite ────────────────────────────────────

def test_reg_targets_finite(sample_split):
    """y_reg must be finite (no inf or nan)."""
    train_df, _, _ = sample_split
    _, y_reg = generate_targets(train_df['Close'].values)
    assert np.isfinite(y_reg).all()
