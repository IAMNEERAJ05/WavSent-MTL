# WavSent-MTL — Task Tracker

## How To Use
- Change [ ] to [x] when done
- Never skip a task — each depends on previous
- Push to GitHub after each session
- If Kaggle session dies, check partial CSVs and
  restart from last unchecked task

---

## PHASE 1 — Data Preparation (Local PC)

### Setup
- [ ] 1.1  Create full folder structure
           (see FOLDER_STRUCTURE.md)
- [ ] 1.2  Initialize GitHub repo, push structure
- [ ] 1.3  Create requirements.txt
- [ ] 1.4  Create config/config.py
- [ ] 1.5  Copy finbert outputs from WSMTE project:
           kotekar_sentiment.csv
           kaggle1_polarity.csv
           kaggle2_polarity.csv
           → data/finbert_outputs/
- [ ] 1.6  Verify kotekar_sentiment.csv columns:
           date, company, polarity_mean
- [ ] 1.7  Verify kaggle1_polarity.csv columns:
           date, polarity_mean, polarity_max
- [ ] 1.8  Verify kaggle2_polarity.csv columns:
           date, polarity_mean, polarity_max

### Notebook 01 — Kotekar Data Prep (Local)
- [ ] 1.9  Download Nifty50 OHLCV via yfinance
           Jan 2017 – May 2024
           → data/raw/nifty50_ohlcv.csv
- [ ] 1.10 Merge kotekar sentiment with price data
- [ ] 1.11 Verify missing dates filled polarity_mean=0
- [ ] 1.12 Save → data/processed/kotekar/merged_data.csv
- [ ] 1.13 Push to GitHub

### Notebook 02 — Kaggle Data Prep (Local)
- [ ] 1.14 Merge kaggle1 + kaggle2 sentiment
- [ ] 1.15 Merge with price data
- [ ] 1.16 Verify gap May–Dec 2021:
           polarity_mean=0, polarity_max=0
- [ ] 1.17 Save → data/processed/kaggle/merged_data.csv
- [ ] 1.18 Push to GitHub

### Notebook 03 — Feature Engineering (Local)
- [ ] 1.19 Apply Coif3 denoising to OHLCV
           (both datasets)
- [ ] 1.20 Compute all 15 candidate features
           on denoised prices (both datasets)
- [ ] 1.21 Verify all features computed correctly
           (spot check RSI range 0-100,
            MACD sign matches price direction)
- [ ] 1.22 Save → data/processed/{dataset}/
           featured_data.csv
- [ ] 1.23 Push to GitHub

### Notebook 04 — Feature Selection (Local)
- [ ] 1.24 70/15/15 split on Kotekar featured_data
- [ ] 1.25 MI ranking on Kotekar training set
           → keep top 10
- [ ] 1.26 Train lightweight LSTM (32 units, 10 seeds)
           on top 10 features
- [ ] 1.27 Compute SHAP → keep top 6-7
- [ ] 1.28 Save selected_features.json (kotekar)
- [ ] 1.29 Apply same features to Kaggle
- [ ] 1.30 Save selected_features.json (kaggle)
- [ ] 1.31 Run full pipeline (Steps 9-16) for kotekar:
           scale, window, targets, class weights
           save all .npy arrays
- [ ] 1.32 Run full pipeline (Steps 9-16) for kaggle:
           scale, window, targets, class weights
           save all .npy arrays
- [ ] 1.33 Verify array shapes match expected
- [ ] 1.34 Run pytest tests/ -v → all pass
- [ ] 1.35 Push all to GitHub

---

## PHASE 2 — Hyperparameter Tuning (Kaggle GPU)

### Notebook 05 — Random Search (Kaggle T4 2x)
- [ ] 2.1  Clone repo in Kaggle
- [ ] 2.2  Upload kotekar processed arrays as
           Kaggle dataset: wavsent-kotekar-processed
- [ ] 2.3  Run random search for TKAN (40 trials)
           → save best_params_tkan.json
- [ ] 2.4  Run random search for LSTM (40 trials)
           → save best_params_lstm.json
- [ ] 2.5  Run random search for GRU (40 trials)
           → save best_params_gru.json
- [ ] 2.6  Run random search for TCN (40 trials)
           → save best_params_tcn.json
- [ ] 2.7  Download all 4 best_params json files
- [ ] 2.8  Update config/config.py with best params
- [ ] 2.9  Push to GitHub

---

## PHASE 3 — Training (Kaggle GPU)

### Notebook 06 — Kotekar Ablation (Kaggle T4 2x)
- [ ] 3.1  Pull latest repo in Kaggle
- [ ] 3.2  Run Config A — 30 seeds — save results
- [ ] 3.3  Run Config B — 30 seeds — save results
- [ ] 3.4  Run Config C — 30 seeds — save results
- [ ] 3.5  Compare B vs C on mean val accuracy
- [ ] 3.6  Update BEST_REPR in config.py
- [ ] 3.7  Run Config D (LSTM) — 30 seeds
- [ ] 3.8  Run Config E (GRU) — 30 seeds
- [ ] 3.9  Run Config F (TCN) — 30 seeds
- [ ] 3.10 Save best model weights per config (.pt)
- [ ] 3.11 Download kotekar_ablation_partial.csv
- [ ] 3.12 Push to GitHub

### Notebook 07 — Kaggle Ablation (Kaggle T4 2x)
- [ ] 3.13 Upload kaggle processed arrays as
           Kaggle dataset: wavsent-kaggle-processed
- [ ] 3.14 Run Configs A–F on Kaggle dataset
           (same structure as notebook 06)
- [ ] 3.15 Save kaggle_ablation_partial.csv
- [ ] 3.16 Push to GitHub

### Notebook 08 — PSO Ensemble (Kaggle T4 2x)
- [ ] 3.17 Load best seed val predictions from
           configs C/D/E/F (or B/D/E/F) per dataset
- [ ] 3.18 Run PSO weight search on kotekar val preds
           → save pso_weights_kotekar.json
- [ ] 3.19 Apply PSO weights to kotekar test preds
           → save Config G kotekar metrics
- [ ] 3.20 Run PSO weight search on kaggle val preds
           → save pso_weights_kaggle.json
- [ ] 3.21 Apply PSO weights to kaggle test preds
           → save Config G kaggle metrics
- [ ] 3.22 Download all results and weights
- [ ] 3.23 Push to GitHub

---

## PHASE 4 — Evaluation (Local PC)

### Notebook 09 — Full Evaluation (Local)
- [ ] 4.1  Merge all ablation results per dataset
- [ ] 4.2  Compute mean, max, std per config
- [ ] 4.3  Run Wilcoxon test (Config A vs Config G)
- [ ] 4.4  Run Shapiro-Wilk normality check
- [ ] 4.5  Generate ablation comparison plots
           (kotekar + kaggle)
- [ ] 4.6  Generate confusion matrix (best config)
- [ ] 4.7  Generate AUC-ROC curves
- [ ] 4.8  Generate loss curves
- [ ] 4.9  Run SHAP analysis on best config
           → shap_summary.png
- [ ] 4.10 Run Granger causality tests
           (polarity_mean vs returns, lags 1-5)
- [ ] 4.11 Run trading simulation (long-only)
- [ ] 4.12 Compute Sharpe ratios
- [ ] 4.13 Save all figures → results/figures/
- [ ] 4.14 Save all tables → results/tables/
- [ ] 4.15 Run baselines (SVM + RF)
           → baselines/run_baselines.py
- [ ] 4.16 Final GitHub push — complete clean repo

---

## PHASE 5 — Paper Writing (Parallel from Phase 3)

- [ ] 5.1  Write Introduction + Related Work outline
- [ ] 5.2  Write Methodology section
- [ ] 5.3  Write Results section (fill after Phase 4)
- [ ] 5.4  Write Discussion section
- [ ] 5.5  Write Abstract + Conclusion
- [ ] 5.6  Compile 10-15 citations beyond 3 refs
- [ ] 5.7  Final paper review

---

## BUFFER

- [ ] 6.1  Fix any bugs from evaluation
- [ ] 6.2  Re-run any failed configs
- [ ] 6.3  Optional: dual-rep extension if time permits
- [ ] 6.4  Optional: walk-forward CV if decided
- [ ] 6.5  Final GitHub push
- [ ] 6.6  Verify all context files match
           final implementation