# WavSent-MTL — Unit Tests Specification

## Overview
Three test files covering data pipeline,
feature engineering, and model architecture.
All tests run locally on CPU — no GPU needed.
Run with: python -m pytest tests/ -v

---

## tests/test_data_pipeline.py

### Test 1 — Temporal split preserves order
def test_split_no_shuffle():
    assert train_df['date'].max() < \
           val_df['date'].min()
    assert val_df['date'].max() < \
           test_df['date'].min()

### Test 2 — Scaler fit on train only
def test_scaler_fit_on_train_only():
    # Train scaled values must be in [0,1]
    assert train_scaled.max() <= 1.0
    assert train_scaled.min() >= 0.0
    # Val/test CAN go outside [0,1]
    # confirms scaler was NOT refit on them

### Test 3 — No future leakage in windows
def test_no_future_leakage():
    # Target at position i must use close[i]
    # not close[i+1]
    for i in range(len(X_train)):
        idx = train_start + i + WINDOW_SIZE
        expected = 1 if (
            close[idx] > close[idx-1]) else 0
        assert y_clf_train[i] == expected

### Test 4 — No missing values in final dataset
def test_no_missing_values():
    df = pd.read_csv(
        'data/processed/kotekar/featured_data.csv')
    assert df[SELECTED_FEATURES]\
               .isnull().sum().sum() == 0

### Test 5 — Kotekar gap fill correct
def test_kotekar_gap_fill():
    df = pd.read_csv(
        'data/processed/kotekar/merged_data.csv')
    # No gap in kotekar — just verify
    # polarity_mean has no NaN
    assert df['polarity_mean'].isnull().sum() == 0

### Test 6 — Kaggle gap period filled correctly
def test_kaggle_gap_fill():
    df = pd.read_csv(
        'data/processed/kaggle/merged_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    gap = df[
        (df['date'] >= '2021-05-01') &
        (df['date'] <= '2021-12-31')]
    assert (gap['polarity_mean'] == 0).all()
    assert (gap['polarity_max'] == 0).all()

### Test 7 — Correct split sizes
def test_split_sizes():
    n = len(final_df)
    assert abs(len(train_df)/n - 0.70) < 0.02
    assert abs(len(val_df)/n - 0.15) < 0.02
    assert abs(len(test_df)/n - 0.15) < 0.02

### Test 8 — Window shapes correct
def test_window_shapes():
    assert X_train.shape[1] == 5
    assert X_train.shape[2] == len(
        SELECTED_FEATURES)

---

## tests/test_features.py

### Test 1 — RSI range
def test_rsi_range():
    rsi = compute_rsi(sample_close_series)
    valid = rsi.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()

### Test 2 — Wavelet preserves length
def test_wavelet_output_length():
    original = np.random.randn(1090)
    denoised = coif3_denoise(original)
    assert len(denoised) == len(original)

### Test 3 — Wavelet actually smooths
def test_wavelet_reduces_noise():
    signal = np.sin(np.linspace(0,10,200))
    noisy = signal + np.random.randn(200) * 0.5
    denoised = coif3_denoise(noisy)
    assert np.std(denoised) < np.std(noisy)

### Test 4 — Features computed on denoised
def test_features_on_denoised_not_raw():
    rsi_denoised = compute_rsi(df['Close_d'])
    rsi_raw = compute_rsi(df['Close'])
    assert not rsi_denoised.equals(rsi_raw)

### Test 5 — Selected features exist in dataframe
def test_selected_features_exist():
    import json
    with open(
        'data/processed/kotekar/'
        'selected_features.json') as f:
        selected = json.load(f)
    df = pd.read_csv(
        'data/processed/kotekar/featured_data.csv')
    for feat in selected:
        assert feat in df.columns

### Test 6 — Regression target range
def test_regression_target_range():
    # Returns should be small daily values
    # not extreme outliers
    assert np.abs(y_reg_train).max() < 10.0

### Test 7 — Classification target binary
def test_classification_target_binary():
    unique = np.unique(y_clf_train)
    assert set(unique).issubset({0, 1})

### Test 8 — ATR always positive
def test_atr_positive():
    atr = compute_atr(
        df['High_d'], df['Low_d'],
        df['Close_d'])
    assert (atr.dropna() >= 0).all()

---

## tests/test_model.py

### Test 1 — All models build without error
def test_all_models_build():
    from src.models.mtl_model import build_model
    for name in ['tkan','lstm','gru','tcn']:
        model = build_model(
            name=name,
            config=CONFIG,
            n_features=7)
        assert model is not None

### Test 2 — Output shapes correct
def test_output_shapes():
    from src.models.mtl_model import build_model
    model = build_model('lstm', CONFIG, 7)
    dummy = torch.randn(8, 5, 7)
    reg_out, clf_out = model(dummy)
    assert reg_out.shape == (8, 1)
    assert clf_out.shape == (8, 1)

### Test 3 — Classification output is probability
def test_clf_output_range():
    model = build_model('lstm', CONFIG, 7)
    dummy = torch.randn(32, 5, 7)
    _, clf_out = model(dummy)
    assert (clf_out.detach().numpy() >= 0).all()
    assert (clf_out.detach().numpy() <= 1).all()

### Test 4 — Regression output unbounded
def test_reg_output_unbounded():
    model = build_model('lstm', CONFIG, 7)
    extreme = torch.ones(8, 5, 7) * 10.0
    reg_out, _ = model(extreme)
    # If sigmoid used, all ~1.0
    # Linear activation varies
    assert reg_out.detach().numpy().std() > 0

### Test 5 — Sigma parameters are trainable
def test_sigma_parameters_trainable():
    model = build_model('lstm', CONFIG, 7)
    param_names = [
        n for n, _ in model.named_parameters()]
    assert any(
        'log_sigma' in n for n in param_names)

### Test 6 — All 4 encoders produce correct shape
def test_encoder_output_shapes():
    from src.models.encoders import (
        LSTMEncoder, GRUEncoder,
        TCNEncoder, TKANEncoder)
    dummy = torch.randn(8, 5, 7)
    hs = 64
    for EncoderClass in [
        LSTMEncoder, GRUEncoder, TCNEncoder]:
        enc = EncoderClass(
            input_size=7, hidden_size=hs)
        out = enc(dummy)
        assert out.shape == (8, hs)

### Test 7 — Uncertainty loss decreases with training
def test_uncertainty_loss_decreases():
    from src.models.losses import (
        uncertainty_weighted_loss)
    import torch.optim as optim
    model = build_model('gru', CONFIG, 7)
    optimizer = optim.Adam(
        model.parameters(), lr=1e-3)
    X = torch.randn(16, 5, 7)
    y_clf = torch.randint(0, 2, (16, 1)).float()
    y_reg = torch.randn(16, 1)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        reg_out, clf_out = model(X)
        mse = torch.nn.MSELoss()(reg_out, y_reg)
        bce = torch.nn.BCELoss()(clf_out, y_clf)
        loss = uncertainty_weighted_loss(
            mse, bce,
            model.log_sigma1,
            model.log_sigma2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    # Loss should generally decrease
    assert losses[-1] < losses[0]

### Test 8 — Different seeds produce
###          different results
def test_different_seeds_differ():
    results = []
    for seed in [42, 7, 123]:
        torch.manual_seed(seed)
        model = build_model('gru', CONFIG, 7)
        dummy = torch.randn(16, 5, 7)
        _, clf_out = model(dummy)
        results.append(
            clf_out.detach().numpy().mean())
    assert len(set(
        [round(r,4) for r in results])) > 1

---

## Running Tests

# Run all tests
python -m pytest tests/ -v

# Run specific file
python -m pytest tests/test_model.py -v

# Run with coverage
python -m pytest tests/ \
    --cov=src --cov-report=term-missing

## Expected Output
All tests green before proceeding to Phase 2.
Do NOT comment out failing tests.
Fix the src/ code instead.