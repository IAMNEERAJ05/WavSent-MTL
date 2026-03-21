"""
src/models/encoders.py
=======================
Four encoder classes for the MTL wrapper.
Each encoder takes input [batch, seq_len, n_features] and
outputs [batch, hidden_size].

Responsibilities:
- LSTMEncoder  : nn.LSTM, return final hidden state
- GRUEncoder   : nn.GRU, return final hidden state
- TCNEncoder   : causal dilated residual blocks
- TKANEncoder  : original Genet & Inzirillo TKAN (import from tkan)
"""

import torch
import torch.nn as nn
from typing import Optional
from config.config import CONFIG


class LSTMEncoder(nn.Module):
    """LSTM encoder returning the final hidden state.

    Args:
        input_size:  Number of input features (n_features).
        hidden_size: LSTM hidden units.
        num_layers:  Number of stacked LSTM layers (1 or 2).
        dropout:     Dropout rate applied between layers (only if num_layers>1).

    Example:
        >>> enc = LSTMEncoder(input_size=7, hidden_size=64)
        >>> out = enc(torch.randn(8, 5, 7))
        >>> out.shape
        torch.Size([8, 64])
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, seq_len, n_features]

        Returns:
            [batch, hidden_size] — final hidden state of last layer.
        """
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # last layer's hidden state


class GRUEncoder(nn.Module):
    """GRU encoder returning the final hidden state.

    Args:
        input_size:  Number of input features.
        hidden_size: GRU hidden units.
        num_layers:  Number of stacked GRU layers (1 or 2).
        dropout:     Dropout rate between layers (only if num_layers>1).

    Example:
        >>> enc = GRUEncoder(input_size=7, hidden_size=64)
        >>> out = enc(torch.randn(8, 5, 7))
        >>> out.shape
        torch.Size([8, 64])
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, seq_len, n_features]

        Returns:
            [batch, hidden_size] — final hidden state of last layer.
        """
        _, h_n = self.gru(x)
        return h_n[-1]


class _TCNBlock(nn.Module):
    """Single causal dilated residual block for TCNEncoder."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        self._pad = pad

    def _causal_trim(self, x: torch.Tensor) -> torch.Tensor:
        """Remove future-padding introduced by causal convolution."""
        return x[:, :, :-self._pad] if self._pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self._causal_trim(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self._causal_trim(self.conv2(out)))
        out = self.dropout(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class TCNEncoder(nn.Module):
    """Temporal Convolutional Network encoder with causal dilated residual blocks.

    Output: feature at the final timestep → [batch, hidden_size].

    Args:
        input_size:  Number of input features.
        hidden_size: Number of channels in each TCN level.
        num_levels:  Number of dilated residual blocks (2 or 3).
        kernel_size: Conv kernel size (2 or 3).
        dropout:     Dropout rate within each block.

    Example:
        >>> enc = TCNEncoder(input_size=7, hidden_size=64, num_levels=2, kernel_size=2)
        >>> out = enc(torch.randn(8, 5, 7))
        >>> out.shape
        torch.Size([8, 64])
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_levels: int = 2,
                 kernel_size: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        for i in range(num_levels):
            in_ch = input_size if i == 0 else hidden_size
            dilation = 2 ** i
            layers.append(_TCNBlock(in_ch, hidden_size, kernel_size,
                                    dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, seq_len, n_features]

        Returns:
            [batch, hidden_size] — output at the last timestep.
        """
        # TCN expects [batch, channels, seq_len]
        x = x.permute(0, 2, 1)
        out = self.network(x)
        return out[:, :, -1]  # take final timestep


class TKANEncoder(nn.Module):
    """TKAN encoder using the original Genet & Inzirillo implementation.

    Falls back to LSTMEncoder if tkan package is not installed,
    and prints a warning. Install with:
        pip install git+https://github.com/remigenet/TKAN.git

    Args:
        input_size:   Number of input features.
        hidden_size:  TKAN hidden size.
        dropout:      Dropout rate.
        spline_order: KAN spline order (fixed=3 per DECISIONS.md).

    Example:
        >>> enc = TKANEncoder(input_size=7, hidden_size=64)
        >>> out = enc(torch.randn(8, 5, 7))
        >>> out.shape
        torch.Size([8, 64])
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 dropout: float = 0.0,
                 spline_order: int = 3):
        super().__init__()
        self._using_fallback = False
        try:
            from tkan import TKAN
            self.tkan = TKAN(
                input_size=input_size,
                hidden_size=hidden_size,
                spline_order=spline_order,
                dropout=dropout,
            )
        except ImportError:
            import warnings
            warnings.warn(
                "tkan package not found. Using LSTMEncoder as fallback. "
                "Install: pip install git+https://github.com/remigenet/TKAN.git",
                UserWarning,
                stacklevel=2,
            )
            self._using_fallback = True
            self.fallback = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, seq_len, n_features]

        Returns:
            [batch, hidden_size]
        """
        if self._using_fallback:
            _, (h_n, _) = self.fallback(x)
            return h_n[-1]
        output = self.tkan(x)
        # TKAN returns [batch, seq_len, hidden_size] — take last step
        if output.dim() == 3:
            return output[:, -1, :]
        return output
