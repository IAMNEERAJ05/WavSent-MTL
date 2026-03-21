"""
src/models/mtl_model.py
========================
MTL wrapper combining any encoder with regression + classification heads.
Uncertainty-weighted loss parameters (log_sigma1, log_sigma2) are
trainable nn.Parameters owned by this module.

Responsibilities:
- MTLModel     : wraps encoder + heads, exposes forward() and sigma params
- build_model  : factory function to instantiate any named encoder + MTLModel
"""

import torch
import torch.nn as nn
from typing import Tuple
from config.config import CONFIG
from src.models.encoders import (
    LSTMEncoder, GRUEncoder, TCNEncoder, TKANEncoder
)
from src.models.heads import RegressionHead, ClassificationHead


class MTLModel(nn.Module):
    """Multi-task learning model: encoder + regression head + classification head.

    log_sigma1 and log_sigma2 are learnable parameters for uncertainty weighting.
    Initialized to CONFIG['log_sigma_init'] (0.0).

    Args:
        encoder:     Any encoder module that maps [batch, seq, feat] → [batch, hs].
        hidden_size: Encoder output dimension (must match encoder's hidden_size).

    Example:
        >>> enc = LSTMEncoder(input_size=7, hidden_size=64)
        >>> model = MTLModel(encoder=enc, hidden_size=64)
        >>> reg, clf = model(torch.randn(8, 5, 7))
        >>> reg.shape, clf.shape
        (torch.Size([8, 1]), torch.Size([8, 1]))
    """

    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.reg_head = RegressionHead(hidden_size)
        self.clf_head = ClassificationHead(hidden_size)
        init_val = float(CONFIG['log_sigma_init'])
        self.log_sigma1 = nn.Parameter(torch.tensor(init_val))
        self.log_sigma2 = nn.Parameter(torch.tensor(init_val))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and both heads.

        Args:
            x: [batch, seq_len, n_features]

        Returns:
            Tuple (reg_out, clf_out):
            - reg_out: [batch, 1] — unbounded return magnitude prediction.
            - clf_out: [batch, 1] — P(up) ∈ [0, 1].
        """
        h = self.encoder(x)
        return self.reg_head(h), self.clf_head(h)


def build_model(
    name: str,
    config: dict,
    n_features: int,
) -> MTLModel:
    """Factory: build MTLModel for a named encoder using best_params from config.

    Args:
        name:       One of 'tkan', 'lstm', 'gru', 'tcn'.
        config:     CONFIG dict (from config.config).
        n_features: Number of input features (seq feature dimension).

    Returns:
        MTLModel instance with the specified encoder and heads.

    Example:
        >>> model = build_model('lstm', CONFIG, n_features=7)
        >>> isinstance(model, MTLModel)
        True
    """
    params = config['best_params'][name]
    hs = params['hidden_size']
    dr = params['dropout']

    if name == 'lstm':
        encoder = LSTMEncoder(
            input_size=n_features,
            hidden_size=hs,
            num_layers=params.get('num_layers', 1),
            dropout=dr,
        )
    elif name == 'gru':
        encoder = GRUEncoder(
            input_size=n_features,
            hidden_size=hs,
            num_layers=params.get('num_layers', 1),
            dropout=dr,
        )
    elif name == 'tcn':
        encoder = TCNEncoder(
            input_size=n_features,
            hidden_size=hs,
            num_levels=params.get('num_levels', 2),
            kernel_size=params.get('kernel_size', 2),
            dropout=dr,
        )
    elif name == 'tkan':
        encoder = TKANEncoder(
            input_size=n_features,
            hidden_size=hs,
            dropout=dr,
            spline_order=3,
        )
    else:
        raise ValueError(
            f"Unknown encoder name '{name}'. "
            f"Expected one of: tkan, lstm, gru, tcn."
        )

    return MTLModel(encoder=encoder, hidden_size=hs)
