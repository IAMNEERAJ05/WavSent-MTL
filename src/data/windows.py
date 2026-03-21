"""
src/data/windows.py
====================
Sliding window construction, target generation, and class imbalance check.

CRITICAL: Data is NEVER shuffled. Windows preserve temporal order.

Responsibilities:
- create_windows()         : build [n, window, n_features] arrays
- generate_targets()       : classification + regression targets
- check_class_imbalance()  : compute class weights if ratio > threshold
"""

import numpy as np
import pandas as pd
import json
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Optional, Dict
from config.config import CONFIG


def create_windows(scaled_data: np.ndarray,
                   window_size: int = None) -> np.ndarray:
    """Build sliding windows of shape [n_samples, window_size, n_features].

    No shuffling. Windows are in strict temporal order.
    First (window_size) rows are used as context for the first sample.

    Args:
        scaled_data: 2-D array [n_rows, n_features] of scaled feature values.
        window_size: Length of look-back window. Defaults to CONFIG['window_size'] (5).

    Returns:
        3-D float32 array of shape [n_rows - window_size, window_size, n_features].

    Example:
        >>> X = create_windows(train_scaled)
        >>> X.shape[1] == 5
        True
    """
    if window_size is None:
        window_size = CONFIG['window_size']
    windows = []
    for i in range(window_size, len(scaled_data)):
        windows.append(scaled_data[i - window_size:i])
    return np.array(windows, dtype=np.float32)


def generate_targets(
    close_prices: np.ndarray,
    window_size: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate classification and regression targets aligned with windows.

    For each window ending at index i, the target is based on Close[i]:
    - clf: 1 if Close[i] > Close[i-1] else 0
    - reg: (Close[i] - Close[i-1]) / Close[i-1]

    Args:
        close_prices: 1-D array of raw Close prices (not denoised, not scaled).
                      Length must match the split DataFrame rows.
        window_size:  Defaults to CONFIG['window_size'] (5).

    Returns:
        Tuple (y_clf, y_reg):
        - y_clf: int32 array of shape [n_samples,] with values in {0, 1}.
        - y_reg: float32 array of shape [n_samples,] of daily returns.

    Example:
        >>> y_clf, y_reg = generate_targets(close_array)
        >>> set(np.unique(y_clf)).issubset({0, 1})
        True
    """
    if window_size is None:
        window_size = CONFIG['window_size']
    y_clf, y_reg = [], []
    for i in range(window_size, len(close_prices)):
        direction = 1 if close_prices[i] > close_prices[i - 1] else 0
        ret = (close_prices[i] - close_prices[i - 1]) / close_prices[i - 1]
        y_clf.append(direction)
        y_reg.append(ret)
    return (
        np.array(y_clf, dtype=np.int32),
        np.array(y_reg, dtype=np.float32),
    )


def check_class_imbalance(
    y_clf_train: np.ndarray,
    save_path: Optional[str] = None,
) -> Optional[Dict[int, float]]:
    """Check class imbalance ratio and compute weights if needed.

    Applies class weights to BCE if majority/minority ratio > threshold (1.5).

    Args:
        y_clf_train: 1-D int array of training classification labels.
        save_path:   If provided, save class_weights.json to this path.

    Returns:
        Dict {0: w0, 1: w1} if imbalanced, else None.

    Example:
        >>> cw = check_class_imbalance(y_clf_train, 'class_weights.json')
        >>> cw is None or isinstance(cw, dict)
        True
    """
    threshold = CONFIG['imbalance_threshold']
    counts = Counter(y_clf_train)
    ratio = max(counts.values()) / min(counts.values())

    up_pct = counts[1] / len(y_clf_train) * 100
    down_pct = counts[0] / len(y_clf_train) * 100
    print(f"Up: {up_pct:.1f}% | Down: {down_pct:.1f}% | Ratio: {ratio:.2f}")

    if ratio > threshold:
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=y_clf_train,
        )
        class_weights = {0: float(weights[0]), 1: float(weights[1])}
    else:
        class_weights = None

    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(class_weights, f)
        print(f"Class weights saved to {save_path}: {class_weights}")

    return class_weights


def temporal_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame 70/15/15 in temporal order — no shuffling.

    Args:
        df: Full feature DataFrame with a 'Date' column, temporally sorted.

    Returns:
        Tuple (train_df, val_df, test_df) with verified date ordering.

    Example:
        >>> train, val, test = temporal_split(featured_df)
        >>> train['Date'].max() < val['Date'].min()
        True
    """
    n = len(df)
    train_end = int(n * CONFIG['train_ratio'])
    val_end = int(n * (CONFIG['train_ratio'] + CONFIG['val_ratio']))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    assert train_df['Date'].max() < val_df['Date'].min(), \
        "Train/val date overlap detected"
    assert val_df['Date'].max() < test_df['Date'].min(), \
        "Val/test date overlap detected"

    print(f"Split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    return train_df, val_df, test_df
