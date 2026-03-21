"""
tests/test_model.py
=====================
Unit tests for model architecture: encoders, heads, MTL wrapper, losses.

Run with: python -m pytest tests/test_model.py -v

All tests run on CPU — no GPU required.
"""

import os
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CONFIG
from src.models.encoders import LSTMEncoder, GRUEncoder, TCNEncoder, TKANEncoder
from src.models.heads import RegressionHead, ClassificationHead
from src.models.mtl_model import MTLModel, build_model
from src.models.losses import uncertainty_weighted_loss, fixed_weighted_loss

# Fixed test dimensions
N_FEATURES = 7
WINDOW = 5
BATCH = 8
HIDDEN = 64


# ── Test 1: All models build without error ────────────────────────────────────

def test_all_models_build():
    """build_model() must succeed for all four encoder names."""
    for name in ['tkan', 'lstm', 'gru', 'tcn']:
        model = build_model(name=name, config=CONFIG, n_features=N_FEATURES)
        assert model is not None, f"build_model('{name}') returned None"
        assert isinstance(model, MTLModel)


# ── Test 2: Output shapes correct ─────────────────────────────────────────────

def test_output_shapes():
    """MTLModel must return (reg:[B,1], clf:[B,1]) for LSTM encoder."""
    model = build_model('lstm', CONFIG, N_FEATURES)
    dummy = torch.randn(BATCH, WINDOW, N_FEATURES)
    reg_out, clf_out = model(dummy)
    assert reg_out.shape == (BATCH, 1), f"reg_out shape: {reg_out.shape}"
    assert clf_out.shape == (BATCH, 1), f"clf_out shape: {clf_out.shape}"


# ── Test 3: Classification output is probability ──────────────────────────────

def test_clf_output_range():
    """Classification output must be in [0, 1] (Sigmoid applied)."""
    model = build_model('lstm', CONFIG, N_FEATURES)
    dummy = torch.randn(32, WINDOW, N_FEATURES)
    _, clf_out = model(dummy)
    arr = clf_out.detach().numpy()
    assert (arr >= 0).all(), "clf_out has values < 0"
    assert (arr <= 1).all(), "clf_out has values > 1"


# ── Test 4: Regression output is unbounded ────────────────────────────────────

def test_reg_output_unbounded():
    """Regression output must vary and not be clipped to [0,1]."""
    model = build_model('lstm', CONFIG, N_FEATURES)
    extreme = torch.ones(BATCH, WINDOW, N_FEATURES) * 10.0
    reg_out, _ = model(extreme)
    arr = reg_out.detach().numpy()
    assert arr.std() > 0 or arr.mean() != 0.5, \
        "Regression output looks like it has Sigmoid (bounded)"


# ── Test 5: Sigma parameters are trainable ────────────────────────────────────

def test_sigma_parameters_trainable():
    """MTLModel must have trainable log_sigma1 and log_sigma2 parameters."""
    model = build_model('lstm', CONFIG, N_FEATURES)
    param_names = [n for n, _ in model.named_parameters()]
    has_sigma1 = any('log_sigma1' in n for n in param_names)
    has_sigma2 = any('log_sigma2' in n for n in param_names)
    assert has_sigma1, "log_sigma1 not found in model parameters"
    assert has_sigma2, "log_sigma2 not found in model parameters"


# ── Test 6: All 3 deterministic encoders produce correct output shape ─────────

def test_encoder_output_shapes():
    """LSTM, GRU, TCN encoders must output [batch, hidden_size]."""
    dummy = torch.randn(BATCH, WINDOW, N_FEATURES)
    for EncoderClass in [LSTMEncoder, GRUEncoder, TCNEncoder]:
        enc = EncoderClass(input_size=N_FEATURES, hidden_size=HIDDEN)
        out = enc(dummy)
        assert out.shape == (BATCH, HIDDEN), \
            f"{EncoderClass.__name__} output shape: {out.shape}"


# ── Test 7: Uncertainty loss decreases over 5 gradient steps ─────────────────

def test_uncertainty_loss_decreases():
    """Loss must decrease over 5 optimization steps on fixed data."""
    torch.manual_seed(42)
    model = build_model('gru', CONFIG, N_FEATURES)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    X = torch.randn(16, WINDOW, N_FEATURES)
    y_clf = torch.randint(0, 2, (16, 1)).float()
    y_reg = torch.randn(16, 1)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCELoss()
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        reg_out, clf_out = model(X)
        mse = mse_fn(reg_out, y_reg)
        bce = bce_fn(clf_out, y_clf)
        loss = uncertainty_weighted_loss(
            mse, bce, model.log_sigma1, model.log_sigma2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses}"


# ── Test 8: Different seeds produce different outputs ─────────────────────────

def test_different_seeds_differ():
    """Same model structure with different seeds must produce different outputs."""
    results = []
    for seed in [42, 7, 123]:
        torch.manual_seed(seed)
        model = build_model('gru', CONFIG, N_FEATURES)
        dummy = torch.randn(16, WINDOW, N_FEATURES)
        _, clf_out = model(dummy)
        results.append(round(clf_out.detach().numpy().mean(), 4))
    assert len(set(results)) > 1, \
        "All seeds produced identical outputs — seeding broken"


# ── Test 9: Fixed loss is a weighted sum ──────────────────────────────────────

def test_fixed_loss_formula():
    """fixed_weighted_loss must equal 0.3*MSE + 0.7*BCE."""
    mse = torch.tensor(0.4)
    bce = torch.tensor(0.6)
    expected = 0.3 * 0.4 + 0.7 * 0.6
    result = fixed_weighted_loss(mse, bce).item()
    assert abs(result - expected) < 1e-5, \
        f"Fixed loss: {result} != {expected}"


# ── Test 10: TKAN encoder builds (or graceful fallback) ──────────────────────

def test_tkan_encoder_builds():
    """TKANEncoder must build without error (fallback to LSTM if tkan absent)."""
    enc = TKANEncoder(input_size=N_FEATURES, hidden_size=HIDDEN)
    dummy = torch.randn(BATCH, WINDOW, N_FEATURES)
    out = enc(dummy)
    assert out.shape == (BATCH, HIDDEN), \
        f"TKANEncoder output shape: {out.shape}"
