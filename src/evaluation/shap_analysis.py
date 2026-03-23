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

    # DEBUG: inspect raw SHAP output before any reshaping
    print(f'[SHAP DEBUG] type(shap_values)={type(shap_values)}')
    if isinstance(shap_values, list):
        print(f'[SHAP DEBUG] list len={len(shap_values)}, shap_values[0].shape={np.array(shap_values[0]).shape}')
    else:
        print(f'[SHAP DEBUG] array shape={np.array(shap_values).shape}')

    # GradientExplainer output shape varies by SHAP version / model output:
    #   list of arrays  → [n_outputs][n_explain, window, n_features]  (multi-output)
    #   single ndarray  → [n_explain, window, n_features, 1]          (single sigmoid output)
    #   single ndarray  → [n_explain, window, n_features]             (already squeezed)
    if isinstance(shap_values, list):
        shap_arr = np.array(shap_values[0])   # take first (only) output
    else:
        shap_arr = np.array(shap_values)
        if shap_arr.ndim == 4:
            shap_arr = shap_arr[..., 0]       # squeeze trailing output dim
    # shap_arr: [n_explain, window, n_features]
    print(f'[SHAP DEBUG] shap_arr.shape after reshape={shap_arr.shape}')
    mean_abs_shap = np.abs(shap_arr).mean(axis=1)  # [n_explain, n_features]
    print(f'[SHAP DEBUG] mean_abs_shap.shape={mean_abs_shap.shape}')

    if save_fig:
        fig_dir = os.path.join(CONFIG['figures_dir'], dataset)
        os.makedirs(fig_dir, exist_ok=True)
        save_path = os.path.join(fig_dir, 'shap_summary.png')

        mean_per_feat = mean_abs_shap.mean(axis=0)  # [n_features]
        # ascending order for barh (lowest importance at bottom)
        sorted_idx = np.argsort(mean_per_feat)

        plt.figure(figsize=(10, 6))
        plt.barh(
            [feature_names[i] for i in sorted_idx],
            mean_per_feat[sorted_idx],
        )
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'SHAP Feature Importance — {dataset}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"SHAP summary saved to {save_path}")

    return mean_abs_shap
