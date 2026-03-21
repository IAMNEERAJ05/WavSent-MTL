"""
src/evaluation/shap_analysis.py
================================
SHAP feature importance analysis using GradientExplainer.

Applied to the best config model after training.
Produces per-feature mean absolute SHAP values and summary plot.

Responsibilities:
- run_shap_analysis() : compute SHAP values and save shap_summary.png
"""

import os
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from typing import List, Optional
from config.config import CONFIG
from src.models.mtl_model import MTLModel


class _ClfWrapper(torch.nn.Module):
    """Wrap MTLModel to expose only the classification output for SHAP."""

    def __init__(self, model: MTLModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, clf = self.model(x)
        return clf


def run_shap_analysis(
    model: MTLModel,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    dataset: str,
    n_background: int = 100,
    n_explain: int = 200,
    save_fig: bool = True,
) -> np.ndarray:
    """Compute SHAP values via GradientExplainer and save summary plot.

    Uses the classification head output only (direction prediction task).

    Args:
        model:         Trained MTLModel in eval mode.
        X_train:       [n_train, window, n_features] float32 — for background.
        X_explain:     [n_explain, window, n_features] float32 — to explain.
        feature_names: List of feature names (length = n_features).
        dataset:       'kotekar' or 'kaggle' — for figure save path.
        n_background:  Number of background samples for GradientExplainer.
        n_explain:     Number of samples to compute SHAP for.
        save_fig:      Whether to save shap_summary.png.

    Returns:
        2-D numpy array of shape [n_explain, n_features] with mean-over-window
        absolute SHAP values per feature.

    Example:
        >>> shap_vals = run_shap_analysis(model, X_tr, X_v, feat_names, 'kotekar')
        >>> shap_vals.shape[1] == len(feat_names)
        True
    """
    model.eval()
    clf_model = _ClfWrapper(model)

    background = torch.tensor(X_train[:n_background], dtype=torch.float32)
    explain_data = torch.tensor(X_explain[:n_explain], dtype=torch.float32)

    explainer = shap.GradientExplainer(clf_model, background)
    shap_values = explainer.shap_values(explain_data)

    # shap_values shape: [n_explain, window, n_features]
    shap_arr = np.array(shap_values)
    mean_abs_shap = np.abs(shap_arr).mean(axis=1)  # [n_explain, n_features]

    if save_fig:
        fig_dir = os.path.join(CONFIG['figures_dir'], dataset)
        os.makedirs(fig_dir, exist_ok=True)
        save_path = os.path.join(fig_dir, 'shap_summary.png')

        mean_per_feat = mean_abs_shap.mean(axis=0)  # [n_features]
        sorted_idx = np.argsort(mean_per_feat)[::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(
            [feature_names[i] for i in sorted_idx[::-1]],
            mean_per_feat[sorted_idx[::-1]],
        )
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'SHAP Feature Importance — {dataset}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"SHAP summary saved to {save_path}")

    return mean_abs_shap
