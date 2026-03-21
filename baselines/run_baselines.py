"""
baselines/run_baselines.py
============================
SVM and Random Forest single-task baselines on both datasets.

Baselines use the same selected features as the MTL models.
scikit-learn models are allowed here (outside src/models/).
No shuffling — temporal order preserved.

Usage:
    python baselines/run_baselines.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CONFIG
from src.evaluation.metrics import compute_clf_metrics


def _flatten(X: np.ndarray) -> np.ndarray:
    """Flatten [n_samples, window, n_features] to [n_samples, window*n_features].

    Args:
        X: 3-D array of windowed features.

    Returns:
        2-D array suitable for sklearn models.

    Example:
        >>> X_flat = _flatten(np.zeros((100, 5, 7)))
        >>> X_flat.shape
        (100, 35)
    """
    return X.reshape(X.shape[0], -1)


def run_baselines_on_dataset(dataset: str) -> pd.DataFrame:
    """Run SVM + RF baselines on one dataset and return results DataFrame.

    Args:
        dataset: 'kotekar' or 'kaggle'.

    Returns:
        pd.DataFrame with one row per model with classification metrics.

    Example:
        >>> df = run_baselines_on_dataset('kotekar')
        >>> 'accuracy' in df.columns
        True
    """
    d = CONFIG[f'{dataset}_processed_dir']

    X_train = np.load(d + 'X_train.npy')
    X_val = np.load(d + 'X_val.npy')
    X_test = np.load(d + 'X_test.npy')
    y_clf_train = np.load(d + 'y_clf_train.npy')
    y_clf_test = np.load(d + 'y_clf_test.npy')

    # Combine train+val for baselines (no sequential tuning needed)
    X_tv = np.concatenate([X_train, X_val], axis=0)
    y_tv = np.concatenate([
        y_clf_train,
        np.load(d + 'y_clf_val.npy')
    ], axis=0)

    X_tr_flat = _flatten(X_tv)
    X_te_flat = _flatten(X_test)

    rows = []

    # ── SVM ──────────────────────────────────────────────────────
    print(f"[{dataset}] Training SVM...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
              random_state=42)
    svm.fit(X_tr_flat, y_tv)
    svm_probs = svm.predict_proba(X_te_flat)[:, 1]
    svm_metrics = compute_clf_metrics(y_clf_test, svm_probs)
    rows.append({'model': 'SVM', 'dataset': dataset, **svm_metrics})
    print(f"  SVM test accuracy: {svm_metrics['accuracy']:.4f}")

    # ── Random Forest ─────────────────────────────────────────────
    print(f"[{dataset}] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr_flat, y_tv)
    rf_probs = rf.predict_proba(X_te_flat)[:, 1]
    rf_metrics = compute_clf_metrics(y_clf_test, rf_probs)
    rows.append({'model': 'RF', 'dataset': dataset, **rf_metrics})
    print(f"  RF test accuracy: {rf_metrics['accuracy']:.4f}")

    return pd.DataFrame(rows)


def main():
    """Run baselines on both datasets and save results.

    Args:
        None.

    Returns:
        None.

    Example:
        >>> main()
    """
    os.makedirs('baselines/results', exist_ok=True)

    for dataset in ['kotekar', 'kaggle']:
        print(f"\n{'='*50}")
        print(f"Running baselines on {dataset}")
        print(f"{'='*50}")

        df = run_baselines_on_dataset(dataset)
        save_path = f'baselines/results/{dataset}_baselines.csv'
        df.to_csv(save_path, index=False)
        print(f"\nSaved → {save_path}")
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
