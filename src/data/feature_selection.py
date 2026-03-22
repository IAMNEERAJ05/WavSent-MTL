"""
src/data/feature_selection.py
==============================
Two-stage feature selection: MI ranking then SHAP filter.

Done ONCE on Kotekar training set only.
Same selected features transferred to Kaggle — no re-selection.

Responsibilities:
- mi_ranking()             : rank 15 candidates by mutual information
- shap_filter()            : train lightweight LSTM + compute SHAP, keep top k
- run_feature_selection()  : orchestrate both stages, return final list
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
from sklearn.feature_selection import mutual_info_classif
from typing import List, Tuple
from config.config import CONFIG


def mi_ranking(X_train: np.ndarray,
               y_clf_train: np.ndarray,
               feature_names: List[str]) -> List[str]:
    """Rank features by mutual information with the classification target.

    Args:
        X_train:      2-D array [n_train, n_features] (unscaled or scaled).
        y_clf_train:  1-D array of binary labels {0, 1}.
        feature_names: List of feature names matching columns of X_train.

    Returns:
        List of top-k feature names sorted by MI score descending.
        k = CONFIG['mi_top_k'] (10).

    Example:
        >>> top10 = mi_ranking(X_tr, y_tr, candidate_features)
        >>> len(top10) == 10
        True
    """
    k = CONFIG['mi_top_k']
    seed = CONFIG['feature_selection_seed']
    scores = mutual_info_classif(X_train, y_clf_train, random_state=seed)
    ranking = pd.Series(scores, index=feature_names).sort_values(ascending=False)
    return ranking.head(k).index.tolist()


class _LightweightLSTM(nn.Module):
    """Minimal LSTM for SHAP-based feature importance.

    Not exposed publicly — used only inside shap_filter().
    """

    def __init__(self, n_features: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.sigmoid(self.fc(h_n[-1]))


def shap_filter(X_train: np.ndarray,
                y_clf_train: np.ndarray,
                top_n_final: int = None) -> List[str]:
    """Train lightweight LSTM on top-10 features, compute SHAP, keep top k.

    Trains CONFIG['feature_selection_runs'] seeds and averages SHAP values.

    Args:
        X_train:      2-D array [n_train, n_features] (n_features = top 10).
                      Must be scaled before passing in.
        y_clf_train:  1-D array of binary labels {0, 1}.
        top_n_final:  Number of features to keep. Defaults to CONFIG['shap_top_k'] (7).

    Returns:
        List of indices (int) into the feature dimension of X_train,
        sorted by mean absolute SHAP value descending.

    Example:
        >>> indices = shap_filter(X_top10, y_clf_train, top_n_final=7)
        >>> len(indices) <= 7
        True
    """
    if top_n_final is None:
        top_n_final = CONFIG['shap_top_k']

    n_seeds = CONFIG['feature_selection_runs']
    units = CONFIG['feature_selection_units']
    base_seed = CONFIG['feature_selection_seed']
    window = CONFIG['window_size']
    n_features = X_train.shape[-1]

    # X_train shape: [n_samples, n_features] (2D, before windowing)
    # We create a minimal 3D version [n_samples, window, n_features]
    # using the last `window` rows as a single sequence
    n = len(X_train)
    X_seq = np.stack([
        X_train[max(0, i - window):i] if i >= window
        else np.pad(X_train[:i], ((window - i, 0), (0, 0)))
        for i in range(window, n + 1)
    ], axis=0).astype(np.float32)  # [n-window+1, window, n_features]

    y_seq = y_clf_train[window - 1:].astype(np.float32)

    all_shap = []

    for run in range(n_seeds):
        seed = base_seed + run
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = _LightweightLSTM(n_features=n_features, hidden_size=units)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        X_t = torch.tensor(X_seq)
        y_t = torch.tensor(y_seq).unsqueeze(1)

        model.train()
        for _ in range(20):  # 20 quick epochs
            optimizer.zero_grad()
            out = model(X_t)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()

        model.eval()
        # Use a small background for GradientExplainer
        bg_size = min(50, len(X_t))
        background = X_t[:bg_size]
        explainer = shap.GradientExplainer(model, background)
        shap_vals = explainer.shap_values(X_t[:100])
        # GradientExplainer may return list[array] — unwrap if so
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_arr = np.array(shap_vals)
        # Shape may be (n_samples, window, n_features, 1) — squeeze trailing dim
        if shap_arr.ndim == 4:
            shap_arr = shap_arr[..., 0]
        # shap_arr: (n_samples, window, n_features)
        mean_abs = np.abs(shap_arr).mean(axis=(0, 1))  # [n_features]
        all_shap.append(mean_abs)

    avg_shap = np.mean(all_shap, axis=0)
    top_indices = np.argsort(avg_shap)[::-1][:top_n_final].tolist()
    return top_indices


def run_feature_selection(
    X_train_all: np.ndarray,
    y_clf_train: np.ndarray,
    feature_names: List[str],
    top_n_final: int = None,
) -> Tuple[List[str], List[str]]:
    """Run full two-stage feature selection on Kotekar training data.

    Stage 1: MI ranking → top 10
    Stage 2: SHAP filter → top 6-7

    Args:
        X_train_all:   2-D array [n_train, 15] of all candidate features,
                       already scaled.
        y_clf_train:   1-D binary classification targets.
        feature_names: List of 15 candidate feature names (CONFIG['candidate_features']).
        top_n_final:   Final feature count. Defaults to CONFIG['shap_top_k'] (7).

    Returns:
        Tuple (selected_features, top10_features):
        - selected_features: final top-k feature names after SHAP.
        - top10_features:    intermediate top-10 list after MI.

    Example:
        >>> selected, top10 = run_feature_selection(X, y, cands)
        >>> len(selected) <= 7
        True
    """
    if top_n_final is None:
        top_n_final = CONFIG['shap_top_k']

    # Stage 1 — MI ranking
    top10_names = mi_ranking(X_train_all, y_clf_train, feature_names)
    top10_idx = [feature_names.index(f) for f in top10_names]
    X_top10 = X_train_all[:, top10_idx]

    # Stage 2 — SHAP filter
    shap_indices = shap_filter(X_top10, y_clf_train, top_n_final=top_n_final)
    selected_features = [top10_names[i] for i in shap_indices]

    return selected_features, top10_names
