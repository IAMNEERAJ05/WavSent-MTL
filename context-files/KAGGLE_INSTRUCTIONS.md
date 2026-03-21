# WavSent-MTL — Kaggle Instructions

## Which Notebooks Run Where

| Notebook | Runs On | GPU |
|----------|---------|-----|
| 01_data_prep_kotekar | Local PC | No |
| 02_data_prep_kaggle | Local PC | No |
| 03_feature_engineering | Local PC | No |
| 04_feature_selection | Local PC | No |
| 05_hyperparam_tuning | Kaggle T4 2x | Yes |
| 06_training_kotekar | Kaggle T4 2x | Yes |
| 07_training_kaggle | Kaggle T4 2x | Yes |
| 08_ensemble | Kaggle T4 2x | Yes |
| 09_evaluation | Local PC | No |

---

## One-Time Kaggle Setup

### Enable GPU
Settings (right panel) → Accelerator → GPU T4 x2
Session type → Persistent

### Kaggle Datasets to Create
After Phase 1 completes locally, upload:

Dataset 1: wavsent-kotekar-processed
Upload from data/processed/kotekar/:
- X_train.npy, X_val.npy, X_test.npy
- y_clf_train.npy, y_clf_val.npy, y_clf_test.npy
- y_reg_train.npy, y_reg_val.npy, y_reg_test.npy
- class_weights.json
- selected_features.json
Path in Kaggle: /kaggle/input/wavsent-kotekar-processed/

Dataset 2: wavsent-kaggle-processed
Same structure from data/processed/kaggle/
Path in Kaggle: /kaggle/input/wavsent-kaggle-processed/

---

## Standard Setup Cell (every Kaggle notebook)

!git clone https://github.com/YOUR_USERNAME/WavSent-MTL.git
import sys
sys.path.append('/kaggle/working/WavSent-MTL')

from config.config import CONFIG
import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA available: {torch.cuda.is_available()}")

---

## Notebook 05 — Hyperparameter Tuning

import numpy as np
from config.config import CONFIG
from src.training.hyperparam_tuning import grid_search

# Load kotekar data
X_train = np.load(
    '/kaggle/input/wavsent-kotekar-processed/'
    'X_train.npy')
# ... load all arrays

data = {
    'X_train': X_train, 'X_val': X_val,
    'y_clf_train': y_clf_train,
    'y_clf_val': y_clf_val,
    'y_reg_train': y_reg_train,
    'y_reg_val': y_reg_val,
}

# Run random search per model (40 trials each)
for model_name in ['tkan','lstm','gru','tcn']:
    best = grid_search(
        model_name=model_name,
        data=data,
        config=CONFIG,
        n_trials=40)
    import json
    with open(
        f'/kaggle/working/'
        f'best_params_{model_name}.json','w') as f:
        json.dump(best, f)
    print(f"{model_name} best: {best}")

# Download all 4 json files after completion

---

## Notebook 06 — Kotekar Training

from src.training.trainer import train_multi_run
import pandas as pd

results = pd.DataFrame()

for config_name in ['A','B','C']:
    print(f"Running Config {config_name}...")
    cfg = CONFIG['ablation_configs'][config_name]
    res = train_multi_run(
        config=CONFIG,
        ablation_cfg=cfg,
        config_name=config_name,
        data=data,
        n_runs=30,
        dataset='kotekar')
    results = pd.concat(
        [results, res], ignore_index=True)
    # Save after every config
    results.to_csv(
        '/kaggle/working/'
        'kotekar_ablation_partial.csv',
        index=False)
    print(f"Config {config_name} done.")

# After B vs C comparison:
# Update BEST_REPR in config.py
# Then run D, E, F

for config_name in ['D','E','F']:
    # Same loop as above

results.to_csv(
    '/kaggle/working/kotekar_ablation_AG.csv',
    index=False)

# Download:
# kotekar_ablation_AG.csv
# results/saved_models/kotekar/*.pt

---

## Notebook 07 — Kaggle Training

# Same structure as notebook 06
# Load from /kaggle/input/wavsent-kaggle-processed/
# Save to /kaggle/working/kaggle_ablation_AG.csv

---

## Notebook 08 — PSO Ensemble

from src.ensemble.pso_ensemble import (
    collect_val_predictions,
    run_pso_search,
    apply_ensemble_weights)
import json

# Load saved val predictions from best seed
# of each model (saved during training)

for dataset in ['kotekar', 'kaggle']:
    val_preds = collect_val_predictions(
        dataset=dataset,
        model_names=['tkan','lstm','gru','tcn'],
        config=CONFIG)

    best_weights = run_pso_search(
        val_preds=val_preds,
        val_labels=y_clf_val,
        config=CONFIG)

    print(f"{dataset} PSO weights: {best_weights}")

    with open(
        f'/kaggle/working/'
        f'pso_weights_{dataset}.json','w') as f:
        json.dump(best_weights, f)

    # Apply to test predictions
    test_metrics = apply_ensemble_weights(
        weights=best_weights,
        dataset=dataset,
        config=CONFIG)

    print(f"{dataset} ensemble test metrics:")
    print(test_metrics)

# Download:
# pso_weights_kotekar.json
# pso_weights_kaggle.json
# ensemble_results_kotekar.csv
# ensemble_results_kaggle.csv

---

## Important Reminders

### Before every Kaggle session
- Push latest code to GitHub first
- Re-clone in Kaggle (never use stale cache)
- Verify GPU: Settings → Accelerator → GPU T4 x2

### If Kaggle session dies
- trainer.py saves after EVERY config run
- Check kotekar_ablation_partial.csv
- Re-run only incomplete configs

### After every Kaggle session
- Download ALL output files immediately
  (outputs expire after session ends)
- Push downloaded files to GitHub
- Update TASKS.md checkboxes

### Memory management on Kaggle
- Clear GPU memory between configs:
  torch.cuda.empty_cache()
  import gc; gc.collect()
- If OOM error: reduce batch_size in config.py