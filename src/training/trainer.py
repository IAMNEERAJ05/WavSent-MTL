"""
src/training/trainer.py
========================
Single-run and multi-run (30-seed) training loops.

Each run:
  1. Seeds all RNG sources
  2. Builds model from best_params
  3. Trains with uncertainty-weighted loss + early stopping
  4. Saves val predictions (best seed stored for PSO)
  5. Returns per-run metrics dict

Responsibilities:
- train_single_run()  : train one seed, return metrics + val predictions
- train_multi_run()   : loop 30 seeds, return pd.DataFrame of results
- save_predictions()  : save val/test probability arrays to disk
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, Tuple
from config.config import CONFIG
from src.models.mtl_model import build_model
from src.models.losses import uncertainty_weighted_loss, fixed_weighted_loss
from src.training.early_stopping import EarlyStopping
from src.evaluation.metrics import compute_clf_metrics, compute_reg_metrics


def _seed_everything(seed: int) -> None:
    """Set all RNG seeds for reproducibility.

    Args:
        seed: Integer seed value.

    Returns:
        None.

    Example:
        >>> _seed_everything(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_loaders(
    X_train: np.ndarray,
    y_clf_train: np.ndarray,
    y_reg_train: np.ndarray,
    X_val: np.ndarray,
    y_clf_val: np.ndarray,
    y_reg_val: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders. No shuffling.

    Args:
        X_train, y_clf_train, y_reg_train: Training arrays.
        X_val, y_clf_val, y_reg_val:       Validation arrays.
        batch_size: Batch size for training DataLoader.

    Returns:
        Tuple (train_loader, val_loader).

    Example:
        >>> tr_loader, v_loader = _build_loaders(X_tr, y_c, y_r, X_v, yc_v, yr_v, 32)
    """
    def _to_tensors(*arrays):
        return [torch.tensor(a, dtype=torch.float32) for a in arrays]

    X_t, yc_t, yr_t = _to_tensors(X_train, y_clf_train, y_reg_train)
    X_v, yc_v, yr_v = _to_tensors(X_val, y_clf_val, y_reg_val)

    train_ds = TensorDataset(X_t, yc_t.unsqueeze(1), yr_t.unsqueeze(1))
    val_ds = TensorDataset(X_v, yc_v.unsqueeze(1), yr_v.unsqueeze(1))

    # shuffle=False enforced per DECISIONS.md
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    return train_loader, val_loader


def train_single_run(
    model_name: str,
    n_features: int,
    data: Dict[str, np.ndarray],
    seed: int,
    class_weights: Optional[Dict[int, float]] = None,
    device: str = 'cpu',
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Train one seed of a named model and return metrics + predictions.

    Args:
        model_name:    One of 'tkan', 'lstm', 'gru', 'tcn'.
        n_features:    Number of input features (seq dim).
        data:          Dict with keys: X_train, X_val, X_test,
                       y_clf_train, y_clf_val, y_clf_test,
                       y_reg_train, y_reg_val, y_reg_test.
        seed:          RNG seed for this run.
        class_weights: {0: w0, 1: w1} or None. Applied to BCE if set.
        device:        'cpu' or 'cuda'.

    Returns:
        Tuple (metrics_dict, val_probs, test_probs):
        - metrics_dict: {accuracy, balanced_accuracy, auc, ..., val_accuracy}.
        - val_probs:  [n_val,] float32 probability array.
        - test_probs: [n_test,] float32 probability array.

    Example:
        >>> metrics, vp, tp = train_single_run('lstm', 7, data, seed=42)
        >>> 'accuracy' in metrics
        True
    """
    _seed_everything(seed)
    params = CONFIG['best_params'][model_name]
    lr = params['learning_rate']
    batch_size = params['batch_size']
    loss_type = CONFIG['loss_type']
    max_epochs = CONFIG['max_epochs']
    grad_clip = CONFIG['grad_clip_norm']

    model = build_model(model_name, CONFIG, n_features).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=CONFIG['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=CONFIG['lr_reduce_factor'],
        patience=CONFIG['lr_reduce_patience'],
        min_lr=CONFIG['lr_min'],
    )
    early_stop = EarlyStopping()

    # BCE with optional class weights
    if class_weights is not None:
        w = torch.tensor([class_weights[1]], dtype=torch.float32).to(device)
        bce_fn = nn.BCELoss(weight=w)
    else:
        bce_fn = nn.BCELoss()
    mse_fn = nn.MSELoss()

    train_loader, val_loader = _build_loaders(
        data['X_train'], data['y_clf_train'], data['y_reg_train'],
        data['X_val'], data['y_clf_val'], data['y_reg_val'],
        batch_size=batch_size,
    )

    for epoch in range(max_epochs):
        # ── Train ────────────────────────────────────────────────
        model.train()
        for X_b, yc_b, yr_b in train_loader:
            X_b, yc_b, yr_b = X_b.to(device), yc_b.to(device), yr_b.to(device)
            optimizer.zero_grad()
            reg_out, clf_out = model(X_b)
            mse = mse_fn(reg_out, yr_b)
            bce = bce_fn(clf_out, yc_b)
            if loss_type == 'uncertainty':
                loss = uncertainty_weighted_loss(
                    mse, bce, model.log_sigma1, model.log_sigma2)
            else:
                loss = fixed_weighted_loss(mse, bce)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # ── Validate ─────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            for X_v, yc_v, yr_v in val_loader:
                X_v, yc_v, yr_v = X_v.to(device), yc_v.to(device), yr_v.to(device)
                reg_v, clf_v = model(X_v)
                val_mse = mse_fn(reg_v, yr_v)
                val_bce = bce_fn(clf_v, yc_v)
                if loss_type == 'uncertainty':
                    val_loss = uncertainty_weighted_loss(
                        val_mse, val_bce,
                        model.log_sigma1, model.log_sigma2).item()
                else:
                    val_loss = fixed_weighted_loss(val_mse, val_bce).item()

        scheduler.step(val_loss)
        early_stop(val_loss, model)
        if early_stop.stop:
            break

    # Restore best weights
    early_stop.restore(model)

    # ── Collect predictions ──────────────────────────────────────
    model.eval()
    with torch.no_grad():
        X_v_t = torch.tensor(data['X_val'], dtype=torch.float32).to(device)
        X_te_t = torch.tensor(data['X_test'], dtype=torch.float32).to(device)
        _, clf_val = model(X_v_t)
        _, clf_test = model(X_te_t)
        val_probs = clf_val.squeeze().cpu().numpy()
        test_probs = clf_test.squeeze().cpu().numpy()

    # ── Compute metrics ──────────────────────────────────────────
    val_acc = compute_clf_metrics(
        data['y_clf_val'], val_probs)['accuracy']
    test_metrics = compute_clf_metrics(data['y_clf_test'], test_probs)

    # Val regression metrics
    with torch.no_grad():
        reg_val_out, _ = model(X_v_t)
        val_reg_preds = reg_val_out.squeeze().cpu().numpy()
    reg_metrics = compute_reg_metrics(data['y_reg_val'], val_reg_preds)

    metrics = {**test_metrics, **reg_metrics, 'val_accuracy': val_acc}
    return metrics, val_probs, test_probs


def train_multi_run(
    config_name: str,
    model_name: str,
    n_features: int,
    data: Dict[str, np.ndarray],
    dataset: str,
    class_weights: Optional[Dict[int, float]] = None,
    device: str = 'cpu',
) -> pd.DataFrame:
    """Run 30-seed training and collect results per CONFIG['n_runs'].

    Saves best-seed val predictions to ablation/results/{dataset}/val_predictions/.

    Args:
        config_name:   Ablation config label e.g. 'A', 'B', 'C'.
        model_name:    One of 'tkan', 'lstm', 'gru', 'tcn'.
        n_features:    Input feature count.
        data:          Dict of train/val/test arrays.
        dataset:       'kotekar' or 'kaggle'.
        class_weights: Class weights or None.
        device:        'cpu' or 'cuda'.

    Returns:
        pd.DataFrame with one row per run and columns from CONFIG['results_columns'].

    Example:
        >>> results = train_multi_run('C', 'tkan', 7, data, 'kotekar')
        >>> len(results) == 30
        True
    """
    n_runs = CONFIG['n_runs']
    rows = []
    best_val_acc = -1.0
    best_val_probs = None
    best_test_probs = None

    for run_idx in range(n_runs):
        seed = run_idx  # seeds 0 … 29
        metrics, val_probs, test_probs = train_single_run(
            model_name=model_name,
            n_features=n_features,
            data=data,
            seed=seed,
            class_weights=class_weights,
            device=device,
        )
        row = {
            'config': config_name,
            'model': model_name,
            'seed': seed,
            'run': run_idx,
            'dataset': dataset,
            **metrics,
        }
        rows.append(row)

        if metrics['val_accuracy'] > best_val_acc:
            best_val_acc = metrics['val_accuracy']
            best_val_probs = val_probs
            best_test_probs = test_probs

        print(
            f"Config {config_name} | {model_name} | "
            f"Run {run_idx + 1}/{n_runs} | "
            f"Val acc: {metrics['val_accuracy']:.4f} | "
            f"Test acc: {metrics['accuracy']:.4f}"
        )

    # Save best-seed val predictions for PSO
    save_predictions(
        val_probs=best_val_probs,
        test_probs=best_test_probs,
        model_name=model_name,
        dataset=dataset,
    )

    return pd.DataFrame(rows, columns=CONFIG['results_columns'])


def save_predictions(
    val_probs: np.ndarray,
    test_probs: np.ndarray,
    model_name: str,
    dataset: str,
) -> None:
    """Save val and test prediction probability arrays to disk.

    Saved to: ablation/results/{dataset}/val_predictions/{model_name}_val_preds.npy
              ablation/results/{dataset}/val_predictions/{model_name}_test_preds.npy

    Args:
        val_probs:   [n_val,] float32 probability array.
        test_probs:  [n_test,] float32 probability array.
        model_name:  'tkan', 'lstm', 'gru', or 'tcn'.
        dataset:     'kotekar' or 'kaggle'.

    Returns:
        None.

    Example:
        >>> save_predictions(val_p, test_p, 'tkan', 'kotekar')
    """
    out_dir = os.path.join(
        CONFIG['ablation_dir'], dataset, 'val_predictions'
    )
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, f'{model_name}_val_preds.npy'),
        val_probs.astype(np.float32),
    )
    np.save(
        os.path.join(out_dir, f'{model_name}_test_preds.npy'),
        test_probs.astype(np.float32),
    )
    print(f"Saved {model_name} predictions → {out_dir}")
