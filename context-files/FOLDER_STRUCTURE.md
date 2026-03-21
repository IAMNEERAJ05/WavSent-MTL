# WavSent-MTL — Folder Structure Specification

## CRITICAL INSTRUCTION
Create files EXACTLY as specified here.
Every file has ONE defined responsibility.
Do not merge responsibilities across files.
Do not create files not listed here without
explicit instruction.

---

## Complete Folder Tree

WavSent-MTL/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   └── config.py                        ← ALL hyperparameters,
│                                           paths, flags, column
│                                           names, feature lists
│
├── data/
│   ├── raw/                             ← gitignored
│   │   ├── kotekar_news.csv             ← cols: datePublished,
│   │   │                                   company, symbol,
│   │   │                                   headline, description,
│   │   │                                   articleBody, tags,
│   │   │                                   author, url
│   │   ├── kaggle_news_1.csv            ← cols: Date, Title,
│   │   │                                   URL, sentiment,
│   │   │                                   confidence
│   │   ├── kaggle_news_2.csv            ← cols: Archive, Date,
│   │   │                                   Headline, Headline link
│   │   └── nifty50_ohlcv.csv            ← cols: Date, Open,
│   │                                       High, Low, Close,
│   │                                       Volume
│   │
│   ├── finbert_outputs/                 ← reused from WSMTE
│   │   │                                   pushed to GitHub
│   │   ├── kotekar_sentiment.csv        ← cols: date, 
│   │   │                                   polarity_mean
│   │   ├── kaggle1_polarity.csv         ← cols: date,
│   │   │                                   polarity_mean,
│   │   │                                   polarity_max
│   │   └── kaggle2_polarity.csv         ← cols: date,
│   │                                       polarity_mean,
│   │                                       polarity_max
│   │
│   └── processed/
│       ├── kotekar/
│       │   ├── merged_data.csv          ← price + sentiment
│       │   │                               merged by date
│       │   ├── featured_data.csv        ← after wavelet +
│       │   │                               all 15 features
│       │   ├── selected_features.json   ← top 6-7 features
│       │   │                               after MI + SHAP
│       │   │                               + sentiment cols
│       │   ├── class_weights.json       ← null or
│       │   │                               {0: w0, 1: w1}
│       │   ├── scaler.pkl               ← gitignored
│       │   ├── reg_scaler.pkl           ← gitignored
│       │   ├── X_train.npy              ← gitignored
│       │   ├── X_val.npy                ← gitignored
│       │   ├── X_test.npy               ← gitignored
│       │   ├── y_clf_train.npy          ← gitignored
│       │   ├── y_clf_val.npy            ← gitignored
│       │   ├── y_clf_test.npy           ← gitignored
│       │   ├── y_reg_train.npy          ← gitignored
│       │   ├── y_reg_val.npy            ← gitignored
│       │   └── y_reg_test.npy           ← gitignored
│       │
│       └── kaggle/
│           ├── merged_data.csv
│           ├── featured_data.csv
│           ├── selected_features.json
│           ├── class_weights.json
│           ├── scaler.pkl               ← gitignored
│           ├── reg_scaler.pkl           ← gitignored
│           ├── X_train.npy              ← gitignored
│           ├── X_val.npy                ← gitignored
│           ├── X_test.npy               ← gitignored
│           ├── y_clf_train.npy          ← gitignored
│           ├── y_clf_val.npy            ← gitignored
│           ├── y_clf_test.npy           ← gitignored
│           ├── y_reg_train.npy          ← gitignored
│           ├── y_reg_val.npy            ← gitignored
│           └── y_reg_test.npy           ← gitignored
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                    ← load_price_data()
│   │   │                                   load_kotekar_sentiment()
│   │   │                                   load_kaggle_sentiment()
│   │   │                                   merge_sources()
│   │   ├── preprocessor.py              ← coif3_denoise()
│   │   │                                   apply_scaler()
│   │   │                                   handle_missing()
│   │   ├── feature_engineering.py       ← compute_rsi()
│   │   │                                   compute_macd()
│   │   │                                   compute_bb_width()
│   │   │                                   compute_roc()
│   │   │                                   compute_ema()
│   │   │                                   compute_atr()
│   │   │                                   compute_obv()
│   │   │                                   compute_stoch_k()
│   │   │                                   compute_williams_r()
│   │   │                                   compute_cci()
│   │   │                                   compute_all_features()
│   │   ├── feature_selection.py         ← mi_ranking()
│   │   │                                   shap_filter()
│   │   │                                   run_feature_selection()
│   │   └── windows.py                   ← create_windows()
│   │                                       generate_targets()
│   │                                       check_class_imbalance()
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py                  ← LSTMEncoder
│   │   │                                   GRUEncoder
│   │   │                                   TCNEncoder
│   │   │                                   TKANEncoder
│   │   ├── heads.py                     ← RegressionHead
│   │   │                                   ClassificationHead
│   │   ├── mtl_model.py                 ← MTLModel(encoder,
│   │   │                                   reg_head, clf_head)
│   │   │                                   build_model(name,
│   │   │                                   config, n_features)
│   │   └── losses.py                    ← uncertainty_weighted_loss()
│   │                                       fixed_weighted_loss()
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                   ← train_single_run()
│   │   │                                   train_multi_run()
│   │   │                                   save_predictions()
│   │   ├── hyperparam_tuning.py         ← random_search()
│   │   │                                   evaluate_params()
│   │   └── early_stopping.py            ← EarlyStopping class
│   │
│   ├── ensemble/
│   │   ├── __init__.py
│   │   └── pso_ensemble.py              ← collect_val_predictions()
│   │                                       run_pso_search()
│   │                                       apply_ensemble_weights()
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                   ← compute_clf_metrics()
│       │                                   compute_reg_metrics()
│       │                                   compute_sharpe()
│       ├── shap_analysis.py             ← run_shap_analysis()
│       └── trading_sim.py               ← run_trading_simulation()
│
├── ablation/
│   ├── run_ablation_kotekar.py          ← loops configs A–G
│   │                                       kotekar dataset
│   ├── run_ablation_kaggle.py           ← loops configs A–G
│   │                                       kaggle dataset
│   └── results/
│       ├── kotekar/
│       │   ├── kotekar_ablation.csv     ← one row per run
│       │   └── val_predictions/         ← saved val preds
│       │       ├── tkan_val_preds.npy      per model best seed
│       │       ├── lstm_val_preds.npy
│       │       ├── gru_val_preds.npy
│       │       └── tcn_val_preds.npy
│       └── kaggle/
│           ├── kaggle_ablation.csv
│           └── val_predictions/
│               ├── tkan_val_preds.npy
│               ├── lstm_val_preds.npy
│               ├── gru_val_preds.npy
│               └── tcn_val_preds.npy
│
├── baselines/
│   ├── run_baselines.py                 ← SVM + RF
│   │                                       both datasets
│   └── results/
│       ├── kotekar_baselines.csv
│       └── kaggle_baselines.csv
│
├── notebooks/
│   ├── 01_data_prep_kotekar.ipynb       ← LOCAL
│   ├── 02_data_prep_kaggle.ipynb        ← LOCAL
│   ├── 03_feature_engineering.ipynb     ← LOCAL
│   ├── 04_feature_selection.ipynb       ← LOCAL
│   ├── 05_hyperparam_tuning.ipynb       ← KAGGLE GPU
│   ├── 06_training_kotekar.ipynb        ← KAGGLE GPU
│   ├── 07_training_kaggle.ipynb         ← KAGGLE GPU
│   ├── 08_ensemble.ipynb                ← KAGGLE GPU
│   └── 09_evaluation.ipynb              ← LOCAL
│
├── results/
│   ├── figures/
│   │   ├── feature_selection/
│   │   │   ├── mi_scores.png
│   │   │   └── shap_selection.png
│   │   ├── kotekar/
│   │   │   ├── ablation_comparison.png
│   │   │   ├── confusion_matrix.png
│   │   │   ├── auc_roc_curve.png
│   │   │   ├── loss_curves.png
│   │   │   ├── shap_summary.png
│   │   │   ├── wavelet_denoising.png
│   │   │   └── trading_simulation.png
│   │   └── kaggle/
│   │       ├── ablation_comparison.png
│   │       ├── confusion_matrix.png
│   │       ├── auc_roc_curve.png
│   │       ├── loss_curves.png
│   │       ├── shap_summary.png
│   │       ├── wavelet_denoising.png
│   │       └── trading_simulation.png
│   │
│   ├── tables/
│   │   ├── kotekar/
│   │   │   ├── ablation_summary.csv
│   │   │   ├── granger_results.csv
│   │   │   ├── trading_results.csv
│   │   │   └── pso_weights.json
│   │   └── kaggle/
│   │       ├── ablation_summary.csv
│   │       ├── granger_results.csv
│   │       ├── trading_results.csv
│   │       └── pso_weights.json
│   │
│   └── saved_models/                    ← gitignored
│       ├── kotekar/
│       │   ├── tkan_best.pt
│       │   ├── lstm_best.pt
│       │   ├── gru_best.pt
│       │   └── tcn_best.pt
│       └── kaggle/
│           ├── tkan_best.pt
│           ├── lstm_best.pt
│           ├── gru_best.pt
│           └── tcn_best.pt
│
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   ├── test_features.py
│   └── test_model.py
│
├── context/                             ← Claude Code context files
│   ├── PROJECT_OVERVIEW.md
│   ├── DECISIONS.md
│   ├── ARCHITECTURE.md
│   ├── DATA_PIPELINE.md
│   ├── FOLDER_STRUCTURE.md
│   ├── TASKS.md
│   ├── TESTS_SPEC.md
│   └── KAGGLE_INSTRUCTIONS.md
│
└── logs/                                ← gitignored
    └── training_logs/

---

## File Responsibilities

| File | Single Responsibility |
|------|-----------------------|
| config/config.py | All hyperparameters + paths + flags |
| src/data/loader.py | Load and merge raw data sources |
| src/data/preprocessor.py | Wavelet denoising and scaling |
| src/data/feature_engineering.py | All 15 technical indicators |
| src/data/feature_selection.py | MI ranking + SHAP filter |
| src/data/windows.py | Sliding windows + targets |
| src/models/encoders.py | TKAN, LSTM, GRU, TCN encoder classes |
| src/models/heads.py | Regression + classification heads |
| src/models/mtl_model.py | MTLModel wrapper + build_model() |
| src/models/losses.py | Uncertainty + fixed loss functions |
| src/training/trainer.py | Multi-seed training loop |
| src/training/hyperparam_tuning.py | Random search |
| src/training/early_stopping.py | EarlyStopping class |
| src/ensemble/pso_ensemble.py | PSO weight search + application |
| src/evaluation/metrics.py | All metrics computation |
| src/evaluation/shap_analysis.py | SHAP feature importance |
| src/evaluation/trading_sim.py | Trading simulation + Sharpe |
| ablation/run_ablation_kotekar.py | Kotekar ablation loop |
| ablation/run_ablation_kaggle.py | Kaggle ablation loop |
| baselines/run_baselines.py | SVM + RF baselines |

---

## GitHub Push vs Gitignore

### Push to GitHub
- All src/ .py files
- All notebooks/ .ipynb files
- config/, requirements.txt, README.md, .gitignore
- data/finbert_outputs/ (all 3 CSV files)
- data/processed/kotekar/merged_data.csv
- data/processed/kotekar/featured_data.csv
- data/processed/kotekar/selected_features.json
- data/processed/kotekar/class_weights.json
- data/processed/kaggle/ (same non-npy files)
- ablation/results/ (all CSV files + val_predictions/)
- baselines/results/
- results/figures/ and results/tables/
- context/ (all markdown files)
- tests/

### Gitignore
data/raw/
data/processed/**/*.npy
data/processed/**/*.pkl
results/saved_models/
logs/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.pt
*.h5