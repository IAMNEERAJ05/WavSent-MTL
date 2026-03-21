"""
src/models/heads.py
====================
Regression and classification heads for the MTL wrapper.

Architecture per DECISIONS.md / ARCHITECTURE.md:
  RegressionHead:     Linear(hs,16) → ReLU → Linear(16,1)  [unbounded]
  ClassificationHead: Linear(hs,16) → ReLU → Linear(16,1) → Sigmoid → [0,1]
"""

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """Regression head predicting next-day return magnitude.

    Linear output — no final activation. Suitable for MSE loss.

    Args:
        hidden_size: Input feature dimension (encoder output size).

    Example:
        >>> head = RegressionHead(hidden_size=64)
        >>> out = head(torch.randn(8, 64))
        >>> out.shape
        torch.Size([8, 1])
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, hidden_size]

        Returns:
            [batch, 1] — unbounded regression output.
        """
        return self.net(x)


class ClassificationHead(nn.Module):
    """Binary classification head predicting P(up) ∈ [0, 1].

    Sigmoid final activation. Suitable for BCELoss.

    Args:
        hidden_size: Input feature dimension (encoder output size).

    Example:
        >>> head = ClassificationHead(hidden_size=64)
        >>> out = head(torch.randn(8, 64))
        >>> (out >= 0).all() and (out <= 1).all()
        True
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, hidden_size]

        Returns:
            [batch, 1] — probability in [0, 1].
        """
        return self.net(x)
