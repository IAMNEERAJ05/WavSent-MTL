"""
src/evaluation/metrics.py
==========================
Compute classification, regression, and Sharpe metrics.

All metrics are computed on raw probability arrays — thresholding at 0.5
is applied internally for classification metrics.

Responsibilities:
- compute_clf_metrics()  : accuracy, balanced_accuracy, AUC, precision, recall, F1
- compute_reg_metrics()  : RMSE, MAE, R²
- compute_sharpe()       : Sharpe ratio from daily return predictions
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config.config import CONFIG


def compute_clf_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute binary classification metrics.

    Args:
        y_true:    1-D int array of true labels {0, 1}.
        y_prob:    1-D float array of predicted probabilities P(up).
        threshold: Decision threshold. Default 0.5.

    Returns:
        Dict with keys: accuracy, balanced_accuracy, auc, precision, recall, f1.

    Example:
        >>> m = compute_clf_metrics(y_true, y_prob)
        >>> 0 <= m['accuracy'] <= 1
        True
    """
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'auc': float(auc),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_reg_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute regression metrics.

    Args:
        y_true: 1-D float array of true regression targets.
        y_pred: 1-D float array of predicted regression values.

    Returns:
        Dict with keys: rmse, mae, r2.

    Example:
        >>> m = compute_reg_metrics(y_true, y_pred)
        >>> 'rmse' in m
        True
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        'rmse': float(np.sqrt(mse)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }


def compute_sharpe(
    daily_returns: np.ndarray,
    risk_free_rate: float = None,
) -> float:
    """Compute annualized Sharpe ratio from daily portfolio returns.

    Sharpe = (mean_daily_return * 252 - risk_free_rate) /
             (std_daily_return * sqrt(252))

    Args:
        daily_returns: 1-D array of daily portfolio returns (decimal).
        risk_free_rate: Annual risk-free rate. Defaults to CONFIG['risk_free_rate'] (0.06).

    Returns:
        Annualized Sharpe ratio as float.

    Example:
        >>> sr = compute_sharpe(np.array([0.001, -0.002, 0.003]))
        >>> isinstance(sr, float)
        True
    """
    if risk_free_rate is None:
        risk_free_rate = CONFIG['risk_free_rate']
    if len(daily_returns) == 0 or np.std(daily_returns) == 0:
        return 0.0
    ann_return = np.mean(daily_returns) * 252
    ann_std = np.std(daily_returns) * np.sqrt(252)
    return float((ann_return - risk_free_rate) / ann_std)


def aggregate_run_metrics(
    results_df,
    metrics: list = None,
) -> dict:
    """Aggregate mean, max, std across 30 seeds for each metric.

    Args:
        results_df: pd.DataFrame with one row per run.
        metrics:    List of metric column names to aggregate.
                    Defaults to all numeric metrics in results_columns.

    Returns:
        Dict of {metric: {mean, max, std}} for each metric.

    Example:
        >>> agg = aggregate_run_metrics(df)
        >>> 'accuracy' in agg
        True
    """
    if metrics is None:
        metrics = ['accuracy', 'balanced_accuracy', 'auc',
                   'precision', 'recall', 'f1',
                   'rmse', 'mae', 'r2', 'val_accuracy']
    result = {}
    for m in metrics:
        if m in results_df.columns:
            result[m] = {
                'mean': float(results_df[m].mean()),
                'max': float(results_df[m].max()),
                'std': float(results_df[m].std()),
            }
    return result
