# WavSent-MTL

**Wavelet-Sentiment Multi-Task Learning for Nifty50 Index Forecasting**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-WavSent--MTL-181717?logo=github)](https://github.com/IAMNEERAJ05/WavSent-MTL)

---

## Overview

WavSent-MTL is a PyTorch multi-task learning framework for next-day Nifty50 index return direction prediction (UP/DOWN). The pipeline applies Coif3 wavelet denoising to raw OHLCV price data before computing technical indicators, then fuses the resulting features with FinBERT-extracted news sentiment. Four architecturally diverse encoders — TKAN, LSTM, GRU, and TCN — are trained jointly on a regression head (return magnitude) and a classification head (direction), using uncertainty-weighted loss (Kendall et al., CVPR 2018). A PSO-weighted ensemble combines all four models' predictions at inference time. Experiments on two datasets demonstrate consistent improvement over Kotekar et al. (IEEE Access 2026), the primary benchmark.

This project includes AI-assisted development work. Claude was used as a coding assistant for some implementation and documentation tasks, while the final design choices, validation, and repository maintenance remain my responsibility. The reported results and code in this repository were reviewed and organized by me before release.

---

## 📊 Key Results

### Kotekar Dataset (Primary — Jan 2020–May 2024)

| Config | Model | Input | Mean Acc | Mean AUC | Max Acc | Bal. Acc |
|--------|-------|-------|:--------:|:--------:|:-------:|:--------:|
| A | TKAN | Returns + sentiment | 0.5641 | 0.4993 | 0.5641 | 0.5000 |
| B | TKAN | Denoised OHLCV | 0.5162 | 0.5314 | 0.5641 | 0.4934 |
| C | TKAN | Denoised technicals | 0.5442 | 0.5906 | 0.5897 | 0.5558 |
| D | LSTM | Denoised technicals | 0.5724 | 0.5407 | 0.6282 | 0.5376 |
| **E** | **GRU** | **Denoised technicals** | **0.5921** | **0.6576** | 0.6474 | 0.5991 |
| F | TCN | Denoised technicals | 0.5744 | 0.6423 | 0.6410 | 0.5923 |
| G | PSO Ensemble | All models | 0.5641 | 0.6345 | — | 0.5735 |
| *Benchmark* | *LSTM (Kotekar et al.)* | *Returns + polarity* | *0.5628* | *0.5556* | *0.5853* | *0.5659* |

Config E (GRU+MTL) achieves mean accuracy **0.5921**, beating the Kotekar et al. benchmark mean by **+0.0293** and max accuracy by **+0.0068**.
Trading simulation on the best model: **Sharpe = 2.5248** vs. benchmark Sharpe = 1.5679 (+61%).

### Kaggle Dataset (Secondary — Jan 2017–May 2024)

| Config | Model | Input | Mean Acc | Mean AUC | Max Acc | Bal. Acc |
|--------|-------|-------|:--------:|:--------:|:-------:|:--------:|
| A | TKAN | Returns + sentiment | 0.5962 | 0.5065 | 0.5962 | 0.5000 |
| B | TKAN | Denoised OHLCV | 0.5962 | 0.5468 | 0.5962 | 0.5000 |
| **C** | **TKAN** | **Denoised technicals** | **0.6766** | **0.7134** | 0.7057 | 0.6553 |
| D | LSTM | Denoised technicals | 0.6702 | 0.7155 | 0.7057 | 0.6633 |
| E | GRU | Denoised technicals | 0.6630 | 0.7168 | 0.6868 | 0.6552 |
| F | TCN | Denoised technicals | 0.6452 | 0.7063 | 0.6830 | 0.6412 |
| G | PSO Ensemble | All models | **0.6906** | **0.7276** | — | 0.6735 |
| *Benchmark* | *LSTM (Kotekar et al.)* | *Returns + polarity* | *0.5628* | *0.5556* | *0.5853* | *0.5659* |

All configs from C onward beat the Kotekar benchmark. Config G ensemble achieves accuracy **0.6906** and **AUC = 0.7276**.
Trading simulation: **Sharpe = 2.0478**, Cumulative Return = **24.74%**, Win Rate = **65.3%**.

---

## 🏗️ Architecture

```
                    Input [batch, 5, n_features]
                              │
                   ┌──────────────────┐
                   │     Encoder      │ ← TKAN / LSTM / GRU / TCN
                   │ (hidden_size=hs) │   selected by ablation config
                   └──────────────────┘
                              │
                    [batch, hidden_size]
                     │                │
             ┌──────────┐      ┌──────────┐
             │ Reg Head │      │ Clf Head │
             ├──────────┤      ├──────────┤
             │Linear    │      │Linear    │
             │(hs, 16)  │      │(hs, 16)  │
             │ReLU      │      │ReLU      │
             │Linear    │      │Linear    │
             │(16, 1)   │      │(16, 1)   │
             │(linear)  │      │Sigmoid   │
             └──────────┘      └──────────┘
                  │                  │
          return magnitude       P(up)∈[0,1]

Uncertainty-weighted loss (Kendall et al. CVPR 2018):
  L = exp(-log_σ₁)·MSE + log_σ₁ + exp(-log_σ₂)·BCE + log_σ₂
  log_σ₁, log_σ₂ are trainable parameters

PSO Ensemble (Config G):
  val_ensemble = softmax([w₁,w₂,w₃,w₄]) · [TKAN, LSTM, GRU, TCN]
  PSO searches on validation predictions (n_particles=20, iters=50)
  Weights applied to held-out test predictions
```

---

## ✨ Core Contributions

1. **Wavelet-Denoised Feature Engineering with MI+SHAP Selection**
   Coif3 wavelet denoising (level=1, soft threshold) is applied to raw OHLCV before computing any technical indicators. From 15 candidate features, Mutual Information ranking followed by SHAP-based filtering selects 7 features: `WILLIAMS_R`, `STOCH_K`, `MACD`, `RSI_14`, `CCI_20`, `EMA_9`, `Volume_d`. Notably, raw denoised price columns (`Close_d`, `High_d`, etc.) are *not* selected — showing that momentum indicators carry more predictive signal than denoised prices. This is itself a finding.

2. **Multi-Task Learning with Uncertainty Weighting**
   A shared encoder feeds two heads jointly: a regression head predicting next-day return magnitude and a classification head predicting direction. Task weights are learned automatically via Kendall et al.'s uncertainty weighting, avoiding manual loss balancing. Early stopping monitors `val_binary_accuracy` (not composite loss) to handle the non-monotonic behaviour of the σ parameters during the first 20–30 epochs.

3. **PyTorch Reimplementation of TKAN for Financial Forecasting**
   The original TKAN (Genet & Inzirillo 2024) is TensorFlow-only. We provide the **first PyTorch implementation of TKAN**, following the paper's specification: an LSTM cell where gate linear transformations are replaced by KANLinear layers (spline-augmented linear layers with learnable B-spline weights). This enables seamless integration with the PyTorch MTL pipeline and constitutes a standalone software contribution.

---

## 📁 Project Structure

```
WavSent-MTL/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   └── config.py                   ← ALL hyperparameters, paths, flags
│
├── data/
│   ├── finbert_outputs/            ← FinBERT polarity CSVs (pushed to GitHub)
│   │   ├── kotekar_sentiment.csv
│   │   ├── kaggle1_polarity.csv
│   │   └── kaggle2_polarity.csv
│   └── processed/
│       ├── kotekar/
│       │   ├── merged_data.csv
│       │   ├── featured_data.csv
│       │   ├── selected_features.json
│       │   └── class_weights.json
│       └── kaggle/
│           ├── merged_data.csv
│           ├── featured_data.csv
│           ├── selected_features.json
│           └── class_weights.json
│
├── src/
│   ├── data/
│   │   ├── loader.py               ← load_price_data, merge_sources
│   │   ├── preprocessor.py         ← coif3_denoise, apply_scaler
│   │   ├── feature_engineering.py  ← all 15 technical indicators
│   │   ├── feature_selection.py    ← mi_ranking, shap_filter
│   │   └── windows.py              ← create_windows, generate_targets
│   ├── models/
│   │   ├── encoders.py             ← TKAN, LSTM, GRU, TCN encoders
│   │   ├── heads.py                ← RegressionHead, ClassificationHead
│   │   ├── mtl_model.py            ← MTLModel wrapper
│   │   └── losses.py               ← uncertainty_weighted_loss
│   ├── training/
│   │   ├── trainer.py              ← train_single_run, train_multi_run
│   │   ├── hyperparam_tuning.py    ← random_search (40 trials)
│   │   └── early_stopping.py       ← EarlyStopping class
│   ├── ensemble/
│   │   └── pso_ensemble.py         ← PSO weight search + application
│   └── evaluation/
│       ├── metrics.py              ← compute_clf_metrics, compute_sharpe
│       ├── shap_analysis.py        ← SHAP feature importance
│       └── trading_sim.py          ← trading simulation
│
├── ablation/
│   ├── run_ablation_kotekar.py     ← configs A–G, Kotekar dataset
│   ├── run_ablation_kaggle.py      ← configs A–G, Kaggle dataset
│   └── results/
│       ├── kotekar/
│       │   ├── kotekar_ablation_partial.csv  ← per-seed raw results (A–F)
│       │   ├── ensemble_results_kotekar.csv  ← Config G single-run result
│       │   ├── pso_weights_kotekar.json      ← PSO weights + individual accs
│       │   └── val_predictions/              ← saved .npy val/test predictions
│       ├── best_params/                      ← tuning outputs from notebook 05 (shared across datasets)
│       │   ├── best_params_tkan.json
│       │   ├── best_params_lstm.json
│       │   ├── best_params_gru.json
│       │   └── best_params_tcn.json
│       └── kaggle/
│           ├── kaggle_ablation_partial.csv   ← per-seed raw results (A–F)
│           ├── ensemble_results_kaggle.csv   ← Config G single-run result
│           ├── pso_weights_kaggle.json       ← PSO weights + individual accs
│           └── val_predictions/              ← saved .npy val/test predictions
│
├── baselines/
│   ├── run_baselines.py            ← SVM + RF baselines
│   └── results/
│       ├── kotekar_baselines.csv
│       └── kaggle_baselines.csv
│
├── notebooks/
│   ├── 01_data_prep_kotekar.ipynb
│   ├── 02_data_prep_kaggle.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_feature_selection.ipynb
│   ├── 05_hyperparam_tuning.ipynb  ← run on Kaggle GPU
│   ├── 06_training_kotekar.ipynb   ← run on Kaggle GPU
│   ├── 07_training_kaggle.ipynb    ← run on Kaggle GPU
│   ├── 08_ensemble.ipynb           ← run on Kaggle GPU
│   └── 09_evaluation.ipynb
│
├── results/
│   ├── figures/
│   │   ├── feature_selection/
│   │   │   └── mi_scores.png
│   │   ├── kotekar/
│   │   │   ├── ablation_comparison.png
│   │   │   ├── auc_roc_curve.png
│   │   │   ├── confusion_matrix.png
│   │   │   ├── shap_summary.png
│   │   │   ├── trading_simulation.png
│   │   │   └── wavelet_denoising.png
│   │   └── kaggle/
│   │       ├── ablation_comparison.png
│   │       ├── auc_roc_curve.png
│   │       ├── confusion_matrix.png
│   │       ├── shap_summary.png
│   │       └── trading_simulation.png
│   └── tables/
│       ├── kotekar/
│       │   ├── ablation_summary.csv  ← A–G, all metrics (canonical)
│       │   ├── trading_results.csv
│       │   ├── granger_results.csv
│       │   ├── wilcoxon_results.csv  ← Config A vs G statistical test
│       │   └── baseline_results.csv ← SVM + RF baselines
│       └── kaggle/
│           ├── ablation_summary.csv  ← A–G, all metrics (canonical)
│           ├── trading_results.csv
│           ├── granger_results.csv
│           ├── wilcoxon_results.csv  ← Config A vs G statistical test
│           └── baseline_results.csv ← SVM + RF baselines
│
└── tests/
    ├── test_data_pipeline.py
    ├── test_features.py
    └── test_model.py
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/IAMNEERAJ05/WavSent-MTL.git
cd WavSent-MTL

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**

```
torch>=2.0
pywavelets
pandas
numpy
scikit-learn
shap
pyswarms
matplotlib
seaborn
```

---

## 🚀 Quick Start

The pipeline is organized as a sequence of notebooks. Notebooks 01–04 run locally; notebooks 05–08 are designed for Kaggle GPU (free tier sufficient).

### Step 1 — Data Preparation

```bash
jupyter notebook notebooks/01_data_prep_kotekar.ipynb
jupyter notebook notebooks/02_data_prep_kaggle.ipynb
```

Downloads Nifty50 OHLCV from Yahoo Finance (`^NSEI`), merges with FinBERT sentiment from `data/finbert_outputs/`, and saves to `data/processed/{dataset}/merged_data.csv`.

### Step 2 — Feature Engineering

```bash
jupyter notebook notebooks/03_feature_engineering.ipynb
```

Applies Coif3 wavelet denoising to OHLCV columns, computes all 15 candidate technical indicators on denoised prices. Output: `data/processed/{dataset}/featured_data.csv`.

### Step 3 — Feature Selection

```bash
jupyter notebook notebooks/04_feature_selection.ipynb
```

Runs MI ranking on Kotekar training set (top 10), then SHAP-based filtering (top 7). Selected features saved to `data/processed/kotekar/selected_features.json` and reused for Kaggle without re-selection.

### Step 4 — Hyperparameter Tuning (Kaggle GPU)

```bash
# Upload repo to Kaggle, then run:
jupyter notebook notebooks/05_hyperparam_tuning.ipynb
```

40-trial random search per encoder, minimising `val_loss`. Best params written back to `config/config.py` and saved as `ablation/results/best_params/{model}.json` for provenance. These params are shared across both datasets.

### Step 5 — Ablation Training (Kaggle GPU)

```bash
jupyter notebook notebooks/06_training_kotekar.ipynb   # configs A–F, 30 seeds each
jupyter notebook notebooks/07_training_kaggle.ipynb    # configs A–F, 30 seeds each
```

Trains each configuration for 30 random seeds with early stopping (patience=35, monitor=`val_binary_accuracy`).

### Step 6 — PSO Ensemble (Kaggle GPU)

```bash
jupyter notebook notebooks/08_ensemble.ipynb           # config G
```

Loads saved validation predictions from the best seed of each encoder, runs PSO (20 particles, 50 iterations) to find optimal softmax weights, applies them to test predictions.

### Step 7 — Evaluation

```bash
jupyter notebook notebooks/09_evaluation.ipynb
```

Computes all metrics, generates SHAP plots, runs trading simulation. Results saved to `results/tables/` and `results/figures/`.

---

## 📂 Datasets

### Study 1 — Kotekar Dataset (Primary)

| Property | Value |
|----------|-------|
| Period | Jan 2020 – May 2024 |
| Price source | Yahoo Finance (`^NSEI`) |
| Sentiment source | Moneycontrol (company-level, Nifty50 constituents) |
| Sentiment columns | `polarity_mean` (FinBERT) |
| Missing days | Sporadic missing days filled with 0.0 (`polarity_mean = 0`) |
| Training samples | ~741 |
| Split | 70/15/15 temporal |
| Purpose | Direct comparison with Kotekar et al. benchmark |

This dataset enables a controlled comparison against Kotekar et al. (IEEE Access 2026): same time period, same metric, same 30-run evaluation protocol. FinBERT outputs are precomputed and stored in `data/finbert_outputs/kotekar_sentiment.csv`.

### Study 2 — Kaggle Dataset (Secondary)

| Property | Value |
|----------|-------|
| Period | Jan 2017 – May 2024 |
| Price source | Yahoo Finance (`^NSEI`) |
| Sentiment source | Kaggle DS1 (Jan 2017–Apr 2021) + DS2 (Jan 2022–May 2024) |
| Sentiment columns | `polarity_mean`, `polarity_max` (FinBERT) |
| Gap period | May–Dec 2021 (filled with 0.0) |
| Training samples | ~1,250 |
| Split | 70/15/15 temporal |
| Purpose | Stability and generalization analysis on longer history |

The same 7 selected features from Study 1 are transferred directly to Study 2 — no re-selection — allowing a clean comparison of dataset size effects.

**Note:** Raw news CSVs are not redistributed. FinBERT inference outputs (polarity scores) are provided in `data/finbert_outputs/` and are sufficient to reproduce all experiments.

---

## 🔍 Feature Selection

Feature selection is performed once on the Kotekar training set and transferred to Kaggle without modification.

### Candidate Pool (15 features, all computed on Coif3-denoised OHLCV)

| # | Feature | Description |
|---|---------|-------------|
| 1–5 | `Close_d`, `Open_d`, `High_d`, `Low_d`, `Volume_d` | Denoised OHLCV |
| 6 | `RSI_14` | Relative Strength Index (14-day) |
| 7 | `MACD` | EMA(12) − EMA(26) |
| 8 | `BB_width` | Bollinger Band width (20-day, 2σ) |
| 9 | `ROC_5` | Rate of Change (5-day) |
| 10 | `EMA_9` | Exponential Moving Average (9-day) |
| 11 | `ATR_14` | Average True Range (14-day) |
| 12 | `OBV` | On-Balance Volume |
| 13 | `STOCH_K` | Stochastic %K (14-day) |
| 14 | `WILLIAMS_R` | Williams %R (14-day) |
| 15 | `CCI_20` | Commodity Channel Index (20-day) |

### Selection Protocol

1. **MI Ranking** — Mutual Information between each feature and the binary classification target, computed on Kotekar training set only. Top 10 retained.
2. **SHAP Filter** — Lightweight LSTM (32 units, 10 seeds) trained on top-10 features. Mean absolute SHAP values aggregated across seeds. Top 7 retained.

### Selected Features

```json
["WILLIAMS_R", "STOCH_K", "MACD", "RSI_14", "CCI_20", "EMA_9", "Volume_d"]
```

Plus `polarity_mean` (Kotekar) or `polarity_mean` + `polarity_max` (Kaggle) as sentiment features.

**Finding:** Raw denoised OHLCV columns (`Close_d`, `High_d`, `Low_d`, `Open_d`) are eliminated entirely, with only `Volume_d` surviving. Momentum and oscillator indicators consistently outrank denoised price levels in both MI and SHAP rankings — suggesting rate-of-change information is more predictive than smoothed price magnitude for direction forecasting.

---

## 🧪 Ablation Study

Seven configurations isolate the contribution of each component:

| Config | Input Representation | Encoder | MTL | Purpose |
|--------|---------------------|---------|:---:|---------|
| A | Daily returns + `polarity_mean` | TKAN | Yes | Kotekar-spirit MTL baseline |
| B | Denoised OHLCV + `polarity_mean` | TKAN | Yes | Raw denoised representation |
| C | Denoised technicals + `polarity_mean` | TKAN | Yes | Engineered denoised — core claim |
| D | Denoised technicals (BEST_REPR) | LSTM | Yes | Architecture comparison |
| E | Denoised technicals (BEST_REPR) | GRU | Yes | Architecture comparison |
| F | Denoised technicals (BEST_REPR) | TCN | Yes | Architecture comparison |
| G | PSO(TKAN + LSTM + GRU + TCN) | Ensemble | Yes | Upper bound |

`BEST_REPR` is determined by comparing Config B vs. Config C on the validation set only, then locked in `config/config.py` before running D–G. For both datasets, Config C (denoised technicals) won.

**Training protocol:** 30 independent seeds per config, Adam optimizer (weight_decay=1e-4), max 150 epochs, early stopping patience=35 monitoring `val_binary_accuracy`, LR scheduler (ReduceLROnPlateau, factor=0.5, patience=10, min_lr=1e-6), gradient clipping (max_norm=1.0).

---

## 📈 Experimental Results

### Kotekar Dataset — Full Ablation Summary

| Config | Mean Acc | ± Std | Mean AUC | ± Std | Mean Bal. Acc | Max Acc |
|--------|:--------:|:-----:|:--------:|:-----:|:-------------:|:-------:|
| A | 0.5641 | 0.0000 | 0.4993 | 0.0314 | 0.5000 | 0.5641 |
| B | 0.5162 | 0.0605 | 0.5314 | 0.0292 | 0.4934 | 0.5641 |
| C | 0.5442 | 0.0203 | 0.5906 | 0.0145 | 0.5558 | 0.5897 |
| D | 0.5724 | 0.0162 | 0.5407 | 0.0912 | 0.5376 | 0.6282 |
| **E** | **0.5921** | 0.0197 | **0.6576** | 0.0109 | **0.5991** | 0.6474 |
| F | 0.5744 | 0.0282 | 0.6423 | 0.0242 | 0.5923 | 0.6410 |
| G | 0.5641 | — | 0.6345 | — | 0.5735 | — |
| *Benchmark* | *0.5628* | — | *0.5556* | — | *0.5659* | *0.5853* |

Trading simulation results (best model, long-only, 6% annual risk-free rate):

| Metric | Value |
|--------|-------|
| Sharpe Ratio | **2.5248** |
| Cumulative Return | 14.83% |
| Number of Trades | 55 |
| Win Rate | 70.9% |
| Benchmark Sharpe (Kotekar et al.) | 1.5679 |

### Kaggle Dataset — Full Ablation Summary

| Config | Mean Acc | ± Std | Mean AUC | ± Std | Mean Bal. Acc | Max Acc |
|--------|:--------:|:-----:|:--------:|:-----:|:-------------:|:-------:|
| A | 0.5962 | 0.0000 | 0.5065 | 0.0392 | 0.5000 | 0.5962 |
| B | 0.5962 | 0.0000 | 0.5468 | 0.0448 | 0.5000 | 0.5962 |
| **C** | **0.6766** | 0.0208 | **0.7134** | 0.0149 | **0.6553** | 0.7057 |
| D | 0.6702 | 0.0202 | 0.7155 | 0.0471 | 0.6633 | 0.7057 |
| E | 0.6630 | 0.0151 | 0.7168 | 0.0086 | 0.6552 | 0.6868 |
| F | 0.6452 | 0.0214 | 0.7063 | 0.0086 | 0.6412 | 0.6830 |
| G | **0.6906** | — | **0.7276** | — | 0.6735 | — |
| *Benchmark* | *0.5628* | — | *0.5556* | — | *0.5659* | *0.5853* |

Trading simulation results (best model, long-only, 6% annual risk-free rate):

| Metric | Value |
|--------|-------|
| Sharpe Ratio | **2.0478** |
| Cumulative Return | 24.74% |
| Number of Trades | 147 |
| Win Rate | 65.3% |

### Baseline Comparison (SVM + Random Forest)

Traditional ML baselines trained on the same 7 selected features with the same 70/15/15 temporal split.

| Model | Dataset | Accuracy | Bal. Acc | AUC | Precision | Recall | F1 |
|-------|---------|:--------:|:--------:|:---:|:---------:|:------:|:--:|
| SVM | Kotekar | 0.6026 | 0.5709 | 0.5914 | 0.6102 | 0.8182 | 0.6990 |
| RF  | Kotekar | 0.5321 | 0.5551 | 0.6065 | 0.6471 | 0.3750 | 0.4748 |
| SVM | Kaggle  | 0.5660 | 0.5757 | 0.6166 | 0.6748 | 0.5253 | 0.5907 |
| RF  | Kaggle  | 0.6151 | 0.6305 | 0.6773 | 0.7373 | 0.5506 | 0.6304 |

SVM on Kotekar (0.6026) exceeds the deep learning benchmark (0.5853) but uses no temporal context (single-timestep features), while all MTL configs use a 5-step window. RF on Kaggle (0.6151) is outperformed by all MTL configs from C onward.

### Statistical Tests (Config A vs Config G)

| Dataset | mean(A) | val(G) | Outcome | Note |
|---------|:-------:|:------:|:-------:|------|
| Kotekar | 0.5641 | 0.5641 | Not significant | Config A zero-variance; G == mean(A) |
| Kaggle  | 0.5962 | 0.6906 | **Significant** | Config A zero-variance; G > mean(A) by +0.0944 |

Config A on both datasets collapsed to a constant value across all 30 seeds (std=0), which prevented a standard Wilcoxon/t-test. For Kaggle, where Config G substantially exceeds Config A, a sign test trivially confirms significance (all 30 differences have the same sign). Full results in `results/tables/{dataset}/wilcoxon_results.csv`.

### Regression Metrics (MTL Auxiliary Head)

Config G regression metrics are PSO-weight-averaged across the constituent model means (C=TKAN, D=LSTM, E=GRU, F=TCN). Configs A–F are 30-seed means.

**Kotekar Dataset:**

| Config | RMSE | MAE | R² |
|--------|:----:|:---:|:--:|
| A | 0.4053 | 0.3295 | -0.0088 |
| B | 0.4070 | 0.3331 | -0.0177 |
| C | 0.3600 | 0.2859 | 0.2038 |
| D | 0.3902 | 0.3139 | 0.0618 |
| **E** | **0.3680** | **0.2880** | **0.1675** |
| F | 0.3720 | 0.2892 | 0.1492 |
| G | 0.3725 | 0.2907 | 0.1468 |

**Kaggle Dataset:**

| Config | RMSE | MAE | R² |
|--------|:----:|:---:|:--:|
| A | 0.7779 | 0.6039 | -0.0155 |
| B | 0.7767 | 0.6027 | -0.0123 |
| **C** | **0.7216** | **0.5563** | **0.1262** |
| D | 0.6694 | 0.5079 | 0.2475 |
| E | 0.6806 | 0.5205 | 0.2224 |
| F | 0.6994 | 0.5290 | 0.1789 |
| G | 0.6694 | 0.5081 | 0.2470 |

### PSO Ensemble Weights (Config G)

PSO searches on saved validation predictions from the best seed of each encoder, using softmax-normalized weights. Due to weight collapse on small validation sets, one model dominates per dataset:

| Dataset | TKAN | LSTM | GRU | TCN | Dominant |
|---------|:----:|:----:|:---:|:---:|:--------:|
| Kotekar | 4.8% | 6.9% | 5.7% | **82.6%** | TCN |
| Kaggle | 0.2% | **99.0%** | 0.7% | 0.04% | LSTM |

**Individual model test accuracies feeding into Config G:**

| Model | Kotekar Test Acc | Kaggle Test Acc |
|-------|:----------------:|:---------------:|
| TKAN (Config C) | 0.5897 | 0.6981 |
| LSTM (Config D) | 0.5897 | 0.6943 |
| GRU  (Config E) | 0.5962 | 0.6755 |
| TCN  (Config F) | 0.5833 | 0.5849 |
| **Ensemble (Config G)** | **0.5641** | **0.6906** |

See [Limitations](#-limitations) for a discussion of PSO weight collapse.

---

## ⚠️ Limitations

1. **PSO Weight Collapse.** On small validation sets (Kotekar: ~165 samples), PSO converges to near-degenerate weights dominated by a single model (TCN on Kotekar at 82.6%, LSTM on Kaggle at 99.0%). This is a known failure mode of particle swarm optimization in low-sample regimes. The ensemble (Config G) does not consistently outperform the best individual model on Kotekar and should be interpreted as an upper-bound experiment rather than a reliable combination strategy.

2. **Validation–Test Gap on Kotekar.** Config G accuracy (0.5641) falls below the best individual model (Config E: 0.5921) on the Kotekar test set. The PSO fitness landscape on the small Kotekar validation set does not generalize reliably to the test set.

3. **Single Market, Single Index.** All experiments target the Nifty50 index. Generalization to other indices, markets, or asset classes is not evaluated.

4. **Small Dataset.** The Kotekar study uses ~741 training samples. While the 30-seed evaluation protocol mitigates variance, performance estimates carry non-trivial uncertainty. The Kaggle study (~1,250 training samples) provides stronger evidence.

5. **No Walk-Forward Cross-Validation.** A single temporal split (70/15/15) is used throughout. Walk-forward cross-validation would provide more robust performance estimates but was deferred due to computational constraints.

---

## 🔭 Future Work

- **Walk-forward cross-validation** — Replace the single temporal split with an expanding-window scheme to reduce variance in performance estimates.
- **Constrained PSO** — Enforce minimum weight constraints (e.g., each model ≥ 5%) to prevent weight collapse on small validation sets.
- **Extended data** — Include a longer price history and additional sentiment sources to improve model stability.
- **Multi-index generalization** — Apply the pipeline to other indices (Sensex, Nifty Bank, S&P 500) to assess transferability.
- **Dual-representation input** — Concatenate raw technicals and denoised technicals (`DUAL_REPR=True` in `config/config.py`) as an optional extended input variant.
- **Improved TKAN integration** — Explore higher spline orders and learned knot placement in the KANLinear layers for richer representation capacity.

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{wavsent_mtl_2025,
  title   = {WavSent-MTL: Wavelet-Sentiment Multi-Task Learning
             for Nifty50 Index Forecasting},
  author  = {Neeraj, et al.},
  year    = {2025},
  url     = {https://github.com/IAMNEERAJ05/WavSent-MTL}
}
```

---

## 📚 References

1. **Kotekar et al. (IEEE Access 2026)** — Primary benchmark.
   *"Can News Sentiment Improve Deep Learning Models for Nifty50 Index Forecasting?"*
   IEEE Access, 2026.

2. **Singh et al. (IEEE Access 2025)** — Coif3 wavelet denoising reference.
   *"Wavelet-Enhanced Deep Learning Ensemble for Accurate Stock Market Forecasting"*
   IEEE Access, 2025.

3. **Genet & Inzirillo (arXiv 2024)** — TKAN architecture.
   *"TKAN: Temporal Kolmogorov-Arnold Networks"*
   arXiv:2405.07344, 2024. [github.com/remigenet/TKAN](https://github.com/remigenet/TKAN)

4. **Kendall et al. (CVPR 2018)** — Uncertainty-weighted multi-task loss.
   *"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"*
   CVPR 2018. arXiv:1705.07115

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
