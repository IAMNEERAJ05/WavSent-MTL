# WavSent-MTL — Project Overview

## Project Name
WavSent-MTL: Wavelet-Sentiment Multi-Task Learning
for Nifty50 Index Forecasting

## One-Line Summary
A PyTorch multi-task learning framework combining Coif3
wavelet-denoised technical features with FinBERT sentiment,
trained across TKAN, LSTM, GRU, and TCN encoders with
uncertainty-weighted loss, combined via a PSO-weighted
ensemble for next-day Nifty50 return direction prediction.

---

## Problem Statement
- Primary task: Binary classification — predict whether
  Nifty50 closes UP (1) or DOWN (0) next trading day
- Auxiliary task: Regression — predict next-day return
  magnitude (MTL auxiliary signal)
- Primary benchmark: Kotekar et al. (IEEE Access 2026)
  best accuracy = 0.5853, best Sharpe = 1.5679

---

## Three Core Contributions
1. Coif3 wavelet denoising on raw OHLCV before feature
   computation, with MI + SHAP-based selection from
   15 candidate technical features
2. Multi-task learning with uncertainty weighting
   (Kendall et al. CVPR 2018) jointly optimizing
   return magnitude regression + direction classification
3. PSO-weighted ensemble of four independently trained
   MTL models (TKAN, LSTM, GRU, TCN) using validation
   predictions for weight search — methodological
   improvement over Kotekar who used training predictions

---

## Two Experimental Studies

### Study 1 — Kotekar Dataset (Primary)
- Source: Kotekar GitHub, company-level Moneycontrol news
- Period: Jan 2020 – May 2024 (~1,090 trading days)
- Sentiment: polarity_mean only (FinBERT)
- Purpose: Direct comparison with Kotekar et al. benchmark
- Input shape: [5 × n] where n = selected features + 1

### Study 2 — Kaggle Dataset (Secondary)
- Sources: Kaggle DS1 (Jan 2017–Apr 2021) +
           Kaggle DS2 (Jan 2022–May 2024)
- Sentiment: polarity_mean + polarity_max (FinBERT)
- Gap May–Dec 2021: polarity_mean=0, polarity_max=0
- Purpose: Stability analysis on larger dataset
- Input shape: [5 × n] where n = selected features + 2
- Same selected features from Study 1 transferred directly
- FinBERT outputs reused from prior work — no re-inference

---

## Ablation Design

| Config | Input | Model | MTL | Purpose |
|--------|-------|-------|-----|---------|
| A | Returns + polarity_mean | TKAN | Yes | Kotekar-spirit MTL baseline |
| B | Denoised OHLCV + polarity_mean | TKAN | Yes | Raw denoised representation |
| C | Denoised technicals + polarity_mean | TKAN | Yes | Engineered denoised — core claim |
| D | BEST_REPR | LSTM | Yes | Architecture comparison |
| E | BEST_REPR | GRU | Yes | Architecture comparison |
| F | BEST_REPR | TCN | Yes | Architecture comparison |
| G | PSO(TKAN+LSTM+GRU+TCN) | Ensemble | Yes | Upper bound |

BEST_REPR = winner of B vs C on validation set only.
Locked in config.py before running D/E/F/G.

---

## Reference Papers
1. Kotekar et al. (IEEE Access 2026) — PRIMARY BENCHMARK
   Same dataset, same metric, same 30-run protocol
2. Singh et al. (IEEE Access 2025) — Coif3 wavelet
   denoising on Nifty50, ensemble of LSTM+CNN+TCN
3. Genet & Inzirillo (arXiv 2024) — TKAN architecture

---

## Key Methodological Improvements Over Kotekar
- MTL: regression + classification jointly vs
  classification only
- PSO on validation predictions vs training predictions
- MI + SHAP feature selection vs fixed feature set
- Two dataset studies vs single dataset
- Individual model test metrics stored before ensemble
  enabling direct comparison

---

## Target Publication
- Venue: IEEE conference (ICMLA / ICTAI / ICONIP)
- Length: 8 pages
- Primary comparison: Config G vs Kotekar 0.5853
- Safety net: Sharpe ratio improvement from MTL
  regression head informing trading signals

---

## Important Notes for Claude Code Sessions
- Framework is PyTorch — no TensorFlow or Keras anywhere
- TKAN: use original Genet & Inzirillo PyTorch repo
- config/config.py is single source of truth for ALL params
- Both datasets are separate experiments with separate
  processed/ subfolders
- BEST_REPR flag in config.py must be set after B vs C
  comparison before running D/E/F/G
- Test set is NEVER touched until notebook 09
- Read DECISIONS.md before any implementation choice