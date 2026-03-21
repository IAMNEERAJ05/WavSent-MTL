# WavSent-MTL — Finalized Decisions

## CRITICAL INSTRUCTION
Every decision in this file is FINAL and LOCKED.
Do NOT suggest alternatives.
Do NOT change parameters without explicit instruction.
Do NOT add components not listed here.
Read this file before making ANY implementation choice.

---

## 1. Framework
- PyTorch — NOT TensorFlow, NOT Keras
- TKAN: original Genet & Inzirillo PyTorch repo
  (https://github.com/remigenet/TKAN)
- TCN: pytorch-tcn package or clean custom implementation

---

## 2. Datasets

### Price Data
- Source: Yahoo Finance, ticker ^NSEI
- Columns: Open, High, Low, Close, Volume
- Range: Jan 2017 – May 2024 (covers both studies)

### Kotekar Dataset (Study 1 — Primary)
- Period: Jan 2020 – May 2024
- Type: Company-level (Moneycontrol, Nifty50 companies)
- Sentiment column: polarity_mean only
- Source file: data/finbert_outputs/kotekar_sentiment.csv
  (reused from prior work — do NOT re-run FinBERT)
- Columns: date, company, polarity_mean

### Kaggle Dataset (Study 2 — Secondary)
- DS1: Jan 2017 – Apr 2021, Title col → polarity_mean
  + polarity_max
- DS2: Jan 2022 – May 2024, Headline col → polarity_mean
  + polarity_max
- Gap May 2021 – Dec 2021: polarity_mean=0, polarity_max=0
- Source files: data/finbert_outputs/kaggle1_polarity.csv
               data/finbert_outputs/kaggle2_polarity.csv
  (reused from prior work — do NOT re-run FinBERT)

### Rejected / Not Used
- NifSent50: REJECTED
- Subjectivity / mDeBERTa: NOT USED

---

## 3. Preprocessing

### Wavelet Denoising
- Wavelet: Coif3, level=1, soft threshold
- Applied to: Open, High, Low, Close, Volume independently
- Applied FIRST — before any feature computation
- Library: PyWavelets (pywt)
- Output columns: Open_d, High_d, Low_d, Close_d, Volume_d

### 15 Candidate Technical Features
ALL computed on DENOISED prices only:
1.  Close_d
2.  Open_d
3.  High_d
4.  Low_d
5.  Volume_d
6.  RSI_14
7.  MACD (EMA12 - EMA26)
8.  BB_width ((Upper-Lower)/Middle, 20-day)
9.  ROC_5
10. EMA_9
11. ATR_14
12. OBV
13. STOCH_K
14. WILLIAMS_R
15. CCI_20

### Feature Selection Protocol
- Step 1: MI ranking vs classification target on
  Kotekar training set → keep top 10
- Step 2: Train lightweight LSTM (32 units, 10 seeds)
  on top 10 → compute SHAP → keep top 6-7 by mean
  absolute SHAP value
- Done ONCE on Kotekar training set only
- Same features transferred to Kaggle — no re-selection
- Saved to: data/processed/kotekar/selected_features.json

### Data Split
- 70/15/15 temporal — NO shuffling at any point
- Verified: train dates < val dates < test dates

### Scaling
- MinMaxScaler, per-feature (each column independently)
- Fit on training set ONLY
- Saved to: data/processed/{dataset}/scaler.pkl

### Sliding Window
- Window size: 5 days
- Warmup: drop first 26 trading days (MACD warmup)
- Input shape per sample: [5, n_features]

### Target Variables
- Classification: y = 1 if Close_t+1 > Close_t else 0
- Regression: y = (Close_t+1 - Close_t) / Close_t
  (next-day return magnitude, normalized by scaler)

### Class Imbalance
- Apply class weights to BCE if ratio > 1.5 (60:40)
- Saved to: data/processed/{dataset}/class_weights.json

### B vs C Decision Protocol
- Run configs B and C fully (30 seeds each)
- Compare mean val accuracy across 30 seeds
- Winner determined on VALIDATION set only
- Update BEST_REPR in config.py
- Proceed to D, E, F, G using BEST_REPR input

---

## 4. Model Architecture

### MTL Wrapper (same for all encoders)
- RegressionHead: Linear(hs,16) → ReLU → Linear(16,1)
- ClassificationHead: Linear(hs,16) → ReLU →
  Linear(16,1) → Sigmoid
- log_sigma1, log_sigma2: nn.Parameter, init=0.0

### LSTM Encoder
- nn.LSTM, batch_first=True
- num_layers: from random search (1 or 2)
- Return: final hidden state h_n → [batch, hidden_size]

### GRU Encoder
- nn.GRU, batch_first=True
- num_layers: from random search (1 or 2)
- Return: final hidden state h_n → [batch, hidden_size]

### TCN Encoder
- Causal dilated convolutions
- kernel_size: from random search (2 or 3)
- num_levels: from random search (2 or 3)
- Return: output at final timestep → [batch, hidden_size]

### TKAN Encoder
- Original Genet & Inzirillo implementation
- spline_order=3 (fixed, repo default)
- Return: final output → [batch, hidden_size]

### Excluded Architectures
- Transformer: needs large dataset
- BiLSTM: uses future timesteps, illegal in forecasting
- CNN: insufficient for 5-timestep window
- SimpleRNN: replaced by TCN for better diversity
- No PCGrad, No GradNorm

---

## 5. Loss Functions

### Uncertainty Weighting (PRIMARY)
L = exp(-log_σ1)×MSE + log_σ1 +
    exp(-log_σ2)×BCE + log_σ2
- log_σ1, log_σ2: trainable nn.Parameter, init=0.0

### Fixed Weighting (FALLBACK)
L = 0.3×MSE + 0.7×BCE
- Used only if uncertainty weighting shows std > 0.06
- Run same ablation, report both in paper

---

## 6. Training

### Hyperparameter Random Search
- 40 trials per model, minimize val_loss
- Done ONCE on Kotekar val set using each model
- Same best config applied to all 30 seed runs

### Search Spaces
Common to all:
- hidden_size: [32, 64, 128]
- dropout: [0.1, 0.2, 0.3]
- learning_rate: [1e-3, 5e-4, 1e-4]
- batch_size: [16, 32, 64]

LSTM and GRU only:
- num_layers: [1, 2]

TCN only:
- kernel_size: [2, 3]
- num_levels: [2, 3]

TKAN only:
- spline_order: [2, 3] if repo exposes it, else fixed=3

### Training Protocol
- Optimizer: Adam, weight_decay=1e-4
- Max epochs: 100
- Early stopping: patience=15, monitor=val_loss,
  restore best weights
- LR scheduler: ReduceLROnPlateau(factor=0.5,
  patience=7, min_lr=1e-6)
- Gradient clipping: max_norm=1.0
- 30 seeds per config

---

## 7. PSO Ensemble

### Configuration
- Library: pyswarms
- Models: TKAN + LSTM + GRU + TCN (4 models)
- Weights: [w1,w2,w3,w4], softmax normalized
- Fitness: negative val accuracy (PSO minimizes)
- n_particles=20, iterations=50
- options: c1=0.5, c2=0.3, w=0.9

### Critical Protocol
- PSO searches on VALIDATION predictions ONLY
  (improvement over Kotekar — not training predictions)
- Each model saves val predictions after training
- PSO weight search uses saved val predictions
- Ensemble applied to test predictions
- Individual model test metrics stored BEFORE ensemble

---

## 8. Evaluation

### Metrics
- Classification: Accuracy, Balanced Accuracy, AUC,
  Precision, Recall, F1
- Regression: RMSE, MAE, R²
- Report: mean, max, std across 30 seeds

### Statistical Tests
- Wilcoxon signed-rank: Config A vs Config G
- Normality check: Shapiro-Wilk before t-test
  (follow Singh et al. approach)

### Required Outputs
- SHAP: GradientExplainer on best config
- Trading simulation: long-only, Kotekar Algorithm 1
- Sharpe: (return − 6%) / std daily returns
- Confusion matrix, AUC-ROC, loss curves

### Baselines
- SVM: single-task classification, same features
- RF: single-task classification, same features

---

## 9. Optional Extensions
### Dual-Representation (time permits only)
- Flag: DUAL_REPR=True in config.py (default: False)
- Concatenates raw technicals + denoised technicals
- Tested LAST after all main configs complete
- Not a primary contribution

### Walk-Forward CV
- Decision deferred — to be decided later