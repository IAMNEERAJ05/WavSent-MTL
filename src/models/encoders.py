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
- TKANEncoder  : PyTorch-native TKAN (Genet & Inzirillo 2024, from scratch)
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


class KANLinear(nn.Module):
    """
    Spline-augmented linear layer.
    Replaces nn.Linear inside LSTM gates with:
    output = linear(x) + spline(x)
    where spline uses polynomial basis functions
    as a B-spline approximation.

    This is the KAN component of TKAN — learnable
    nonlinear transformations on edges rather than
    fixed activations on nodes (Liu et al. 2024).
    """
    def __init__(self, in_features: int,
                 out_features: int,
                 num_knots: int = 5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, num_knots) * 0.1
        )
        self.num_knots = num_knots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear component
        linear_out = self.linear(x)
        # Spline component using polynomial basis
        x_norm = torch.tanh(x)  # normalize to [-1, 1]
        basis = torch.stack([
            x_norm ** i for i in range(self.num_knots)
        ], dim=-1)  # [batch, in_features, num_knots]
        spline_out = torch.einsum(
            'bik,oik->bo', basis, self.spline_weight
        )
        return linear_out + spline_out


class TKANCell(nn.Module):
    """
    Single TKAN cell.
    Standard LSTM cell where the combined input-hidden
    linear transformation is replaced by KANLinear.
    Gates: input (i), forget (f), cell (g), output (o).
    """
    def __init__(self, input_size: int,
                 hidden_size: int,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        # KAN replaces the standard linear projection
        # over concatenated [input, hidden]
        self.kan_gates = KANLinear(
            in_features=input_size + hidden_size,
            out_features=4 * hidden_size
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple
    ) -> tuple:
        # x: [batch, input_size]
        # state: (h, c) each [batch, hidden_size]
        h, c = state
        combined = torch.cat([x, h], dim=1)
        combined = self.dropout(combined)
        # KAN-transformed gate projections
        gates = self.kan_gates(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        g = torch.tanh(g)      # cell gate
        o = torch.sigmoid(o)   # output gate
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)


class TKANEncoder(nn.Module):
    """
    PyTorch-native TKAN encoder.

    Implements Genet & Inzirillo (2024) TKAN architecture:
    an LSTM where gate linear transformations are replaced
    by KAN-inspired spline-augmented mappings (KANLinear).

    This is a from-scratch PyTorch reimplementation because
    the original remigenet/TKAN repo is TensorFlow-based
    and incompatible with our PyTorch pipeline.

    Returns the final hidden state: [batch, hidden_size].

    Args:
        input_size:   Number of input features per timestep.
        hidden_size:  Hidden state dimension.
        dropout:      Dropout rate applied inside TKANCell.
        spline_order: Kept for API compatibility (num_knots
                      in KANLinear = spline_order + 2).

    Example:
        >>> enc = TKANEncoder(input_size=8, hidden_size=64)
        >>> out = enc(torch.randn(4, 5, 8))
        >>> out.shape
        torch.Size([4, 64])
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        spline_order: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # num_knots derived from spline_order
        num_knots = spline_order + 2
        self.cell = TKANCell(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        # Override KANLinear num_knots with spline_order
        self.cell.kan_gates = KANLinear(
            in_features=input_size + hidden_size,
            out_features=4 * hidden_size,
            num_knots=num_knots,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, input_size].
        Returns:
            Final hidden state [batch, hidden_size].
        """
        batch = x.size(0)
        device = x.device
        h = torch.zeros(batch, self.hidden_size,
                       device=device)
        c = torch.zeros(batch, self.hidden_size,
                       device=device)
        # Unroll over timesteps (seq_len=5)
        for t in range(x.size(1)):
            h, (h, c) = self.cell(x[:, t, :], (h, c))
        return h  # [batch, hidden_size]
