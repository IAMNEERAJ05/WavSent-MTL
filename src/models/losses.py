"""
src/models/losses.py
=====================
Loss functions for MTL training.

Primary:  Uncertainty-weighted loss (Kendall et al. CVPR 2018)
Fallback: Fixed-weight loss (used if uncertainty weighting shows std > 0.06)

Responsibilities:
- uncertainty_weighted_loss() : learnable sigma weighting
- fixed_weighted_loss()       : static 0.3*MSE + 0.7*BCE
"""

import torch
from config.config import CONFIG


def uncertainty_weighted_loss(
    mse: torch.Tensor,
    bce: torch.Tensor,
    log_sigma1: torch.Tensor,
    log_sigma2: torch.Tensor,
) -> torch.Tensor:
    """Uncertainty-weighted multi-task loss (Kendall et al. CVPR 2018).

    L = exp(-log_s1)*MSE + log_s1 + exp(-log_s2)*BCE + log_s2

    log_sigma1 and log_sigma2 are learnable nn.Parameters.
    Larger sigma → task weighted less; the log terms regularize sigma.

    Args:
        mse:       Scalar MSE loss from regression head.
        bce:       Scalar BCE loss from classification head.
        log_sigma1: Learnable log-uncertainty for regression (MTLModel.log_sigma1).
        log_sigma2: Learnable log-uncertainty for classification (MTLModel.log_sigma2).

    Returns:
        Scalar combined loss tensor.

    Example:
        >>> loss = uncertainty_weighted_loss(mse, bce, model.log_sigma1, model.log_sigma2)
        >>> loss.shape
        torch.Size([])
    """
    reg_term = torch.exp(-log_sigma1) * mse + log_sigma1
    clf_term = torch.exp(-log_sigma2) * bce + log_sigma2
    return reg_term + clf_term


def fixed_weighted_loss(
    mse: torch.Tensor,
    bce: torch.Tensor,
) -> torch.Tensor:
    """Fixed-weight multi-task loss: 0.3 * MSE + 0.7 * BCE.

    Fallback used if uncertainty weighting produces std > 0.06 across seeds.
    Weights taken from CONFIG['fixed_mse_weight'] and CONFIG['fixed_bce_weight'].

    Args:
        mse: Scalar MSE loss from regression head.
        bce: Scalar BCE loss from classification head.

    Returns:
        Scalar combined loss tensor.

    Example:
        >>> loss = fixed_weighted_loss(mse, bce)
        >>> loss.shape
        torch.Size([])
    """
    w_mse = CONFIG['fixed_mse_weight']   # 0.3
    w_bce = CONFIG['fixed_bce_weight']   # 0.7
    return w_mse * mse + w_bce * bce
