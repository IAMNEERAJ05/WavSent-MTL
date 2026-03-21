"""
ablation/run_ablation_kotekar.py
==================================
Run full ablation study (Configs A–G) on the Kotekar dataset.

Loops configs in order A → B → C → (manual BEST_REPR update) → D → E → F → G.
Results are saved after EVERY config to prevent data loss.

Usage:
    python ablation/run_ablation_kotekar.py [--configs A B C] [--device cuda]

Data expected at:
    data/processed/kotekar/X_train.npy  (and all other .npy arrays)
    data/processed/kotekar/class_weights.json
    data/processed/kotekar/selected_features.json
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CONFIG
from src.training.trainer import train_multi_run
from src.ensemble.pso_ensemble import (
    collect_val_predictions, run_pso_search, apply_ensemble_weights
)
from src.evaluation.metrics import aggregate_run_metrics

DATASET = 'kotekar'


def load_data() -> dict:
    """Load all processed Kotekar arrays from disk.

    Args:
        None — paths from CONFIG['kotekar_processed_dir'].

    Returns:
        Dict with all train/val/test X, y_clf, y_reg arrays.

    Example:
        >>> data = load_data()
        >>> data['X_train'].shape[1] == 5
        True
    """
    d = CONFIG['kotekar_processed_dir']
    return {
        'X_train':     np.load(d + 'X_train.npy'),
        'X_val':       np.load(d + 'X_val.npy'),
        'X_test':      np.load(d + 'X_test.npy'),
        'y_clf_train': np.load(d + 'y_clf_train.npy'),
        'y_clf_val':   np.load(d + 'y_clf_val.npy'),
        'y_clf_test':  np.load(d + 'y_clf_test.npy'),
        'y_reg_train': np.load(d + 'y_reg_train.npy'),
        'y_reg_val':   np.load(d + 'y_reg_val.npy'),
        'y_reg_test':  np.load(d + 'y_reg_test.npy'),
    }


def load_class_weights():
    """Load class weights from JSON, return dict or None.

    Args:
        None.

    Returns:
        Dict {0: float, 1: float} or None if balanced.

    Example:
        >>> cw = load_class_weights()
    """
    path = CONFIG['kotekar_processed_dir'] + 'class_weights.json'
    with open(path) as f:
        cw = json.load(f)
    if cw is not None:
        return {int(k): v for k, v in cw.items()}
    return None


def resolve_input_type(input_type: str, n_raw: int, n_tech: int) -> int:
    """Resolve BEST_REPR to actual n_features count.

    Args:
        input_type: 'returns_sentiment', 'denoised_ohlcv',
                    'denoised_technicals', or 'BEST_REPR'.
        n_raw:      n_features for denoised_ohlcv input.
        n_tech:     n_features for denoised_technicals input.

    Returns:
        n_features integer for the given input_type.

    Example:
        >>> n = resolve_input_type('BEST_REPR', 6, 8)
        >>> n in (6, 8)
        True
    """
    best = CONFIG['BEST_REPR']
    mapping = {
        'returns_sentiment':   n_tech,  # returns+polarity — same n_feat as tech+sent
        'denoised_ohlcv':      n_raw,
        'denoised_technicals': n_tech,
        'BEST_REPR':           n_raw if best == 'denoised_ohlcv' else n_tech,
    }
    return mapping[input_type]


def main(configs_to_run=None, device='cpu'):
    """Run ablation loop and save results after each config.

    Args:
        configs_to_run: List of config keys to run e.g. ['A','B','C'].
                        Defaults to all A–F (G handled separately).
        device:         'cpu' or 'cuda'.

    Returns:
        None.

    Example:
        >>> main(configs_to_run=['A'], device='cpu')
    """
    data = load_data()
    class_weights = load_class_weights()
    n_features = data['X_train'].shape[2]

    result_dir = os.path.join(CONFIG['ablation_dir'], DATASET)
    os.makedirs(result_dir, exist_ok=True)
    results_path = os.path.join(result_dir, 'kotekar_ablation.csv')

    # Load existing results if resuming
    if os.path.exists(results_path):
        all_results = pd.read_csv(results_path)
        done_configs = set(all_results['config'].unique())
        print(f"Resuming. Already done: {done_configs}")
    else:
        all_results = pd.DataFrame()
        done_configs = set()

    ablation = CONFIG['ablation_configs']
    if configs_to_run is None:
        configs_to_run = [k for k in ablation if k != 'G']

    for cfg_key in configs_to_run:
        if cfg_key in done_configs:
            print(f"Config {cfg_key} already done — skipping.")
            continue

        cfg = ablation[cfg_key]
        model_name = cfg['model']
        input_type = cfg['input_type']

        print(f"\n{'='*60}")
        print(f"Config {cfg_key}: {cfg['description']}")
        print(f"{'='*60}")

        df = train_multi_run(
            config_name=cfg_key,
            model_name=model_name,
            n_features=n_features,
            data=data,
            dataset=DATASET,
            class_weights=class_weights,
            device=device,
        )

        all_results = pd.concat([all_results, df], ignore_index=True)
        all_results.to_csv(results_path, index=False)
        print(f"Config {cfg_key} saved → {results_path}")

        # Print aggregated summary
        agg = aggregate_run_metrics(df)
        print(f"\nConfig {cfg_key} summary:")
        for m, vals in agg.items():
            print(f"  {m}: mean={vals['mean']:.4f} std={vals['std']:.4f}")

    print("\n✓ All specified configs complete.")
    print("  REMINDER: If B and C are both done, compare val accuracy,")
    print("  update CONFIG['BEST_REPR'] in config.py, then run D/E/F.")


def run_config_g(device='cpu'):
    """Run Config G: PSO ensemble of best-seed predictions.

    CRITICAL: Run ONLY after configs C/D/E/F are complete
              and their val/test predictions are saved.

    Args:
        device: Unused (PSO runs on CPU).

    Returns:
        None — saves G metrics to kotekar_ablation.csv.

    Example:
        >>> run_config_g()
    """
    data = load_data()
    result_dir = os.path.join(CONFIG['ablation_dir'], DATASET)
    results_path = os.path.join(result_dir, 'kotekar_ablation.csv')

    val_preds, test_preds = collect_val_predictions(DATASET)

    weights = run_pso_search(val_preds, data['y_clf_val'])

    # Store individual test metrics per model BEFORE ensemble
    print("\nIndividual model test metrics (pre-ensemble):")
    from src.evaluation.metrics import compute_clf_metrics
    for m in CONFIG['pso_models']:
        ind_metrics = compute_clf_metrics(data['y_clf_test'], test_preds[m])
        print(f"  {m}: {ind_metrics}")

    g_metrics = apply_ensemble_weights(weights, test_preds, data['y_clf_test'])

    # Save Config G row
    row = {
        'config': 'G', 'model': 'ensemble', 'seed': 0, 'run': 0,
        'dataset': DATASET, **g_metrics, 'val_accuracy': -1.0,
        'rmse': 0.0, 'mae': 0.0, 'r2': 0.0,
    }
    all_results = pd.read_csv(results_path) if os.path.exists(results_path) \
        else pd.DataFrame()
    all_results = pd.concat([all_results, pd.DataFrame([row])], ignore_index=True)
    all_results.to_csv(results_path, index=False)

    # Save PSO weights
    import json
    weights_path = os.path.join(CONFIG['tables_dir'], DATASET, 'pso_weights.json')
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"PSO weights saved → {weights_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', default=None,
                        help='Config keys to run e.g. A B C')
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--config-g', action='store_true',
                        help='Run Config G (PSO ensemble)')
    args = parser.parse_args()

    if args.config_g:
        run_config_g(device=args.device)
    else:
        main(configs_to_run=args.configs, device=args.device)
