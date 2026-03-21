"""
src/evaluation/trading_sim.py
==============================
Long-only trading simulation following Kotekar et al. Algorithm 1.

Signal: buy when P(up) >= 0.5, hold cash otherwise.
Computes daily portfolio returns and Sharpe ratio.

Responsibilities:
- run_trading_simulation() : simulate trades, return daily returns + Sharpe
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from config.config import CONFIG
from src.evaluation.metrics import compute_sharpe


def run_trading_simulation(
    y_prob: np.ndarray,
    close_prices: np.ndarray,
    dataset: str,
    config_name: str,
    threshold: float = 0.5,
    save_fig: bool = True,
) -> Dict[str, float]:
    """Simulate long-only trading using model probability predictions.

    Strategy (Kotekar Algorithm 1):
    - Day i: if P(up)_i >= threshold → hold long position on day i+1
    - Day i: if P(up)_i < threshold  → hold cash on day i+1
    - Sharpe = (annualized_return - risk_free_rate) / annualized_std

    Args:
        y_prob:       [n_test,] float32 prediction probabilities.
        close_prices: [n_test+1,] raw Close prices for the test period.
                      First price is the "current" price for the first signal.
        dataset:      'kotekar' or 'kaggle' — for figure save path.
        config_name:  Config label (e.g. 'G') — for figure title.
        threshold:    Buy signal threshold. Default 0.5.
        save_fig:     Whether to save trading_simulation.png.

    Returns:
        Dict with keys: sharpe, cumulative_return, n_trades, win_rate.

    Example:
        >>> result = run_trading_simulation(probs, closes, 'kotekar', 'G')
        >>> 'sharpe' in result
        True
    """
    n = len(y_prob)
    signals = (y_prob >= threshold).astype(int)   # 1=long, 0=cash

    # Daily returns: r[i] = (close[i+1] - close[i]) / close[i]
    if len(close_prices) < n + 1:
        close_prices = np.concatenate([[close_prices[0]], close_prices])

    market_returns = np.diff(close_prices[:n + 1]) / close_prices[:n]
    portfolio_returns = signals * market_returns   # long or flat

    sharpe = compute_sharpe(portfolio_returns)
    cumulative = float((1 + portfolio_returns).prod() - 1)
    n_trades = int(signals.sum())
    winning_trades = int((portfolio_returns[signals == 1] > 0).sum())
    win_rate = winning_trades / n_trades if n_trades > 0 else 0.0

    result = {
        'sharpe': sharpe,
        'cumulative_return': cumulative,
        'n_trades': n_trades,
        'win_rate': float(win_rate),
    }

    if save_fig:
        fig_dir = os.path.join(CONFIG['figures_dir'], dataset)
        os.makedirs(fig_dir, exist_ok=True)
        save_path = os.path.join(fig_dir, 'trading_simulation.png')

        cum_portfolio = (1 + portfolio_returns).cumprod()
        cum_market = (1 + market_returns).cumprod()

        plt.figure(figsize=(12, 5))
        plt.plot(cum_portfolio, label=f'Strategy ({config_name})')
        plt.plot(cum_market, label='Buy & Hold', linestyle='--', alpha=0.7)
        plt.xlabel('Trading Days (Test Set)')
        plt.ylabel('Cumulative Return')
        plt.title(f'Trading Simulation — {dataset} — Config {config_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Trading simulation plot saved to {save_path}")

    print(
        f"Sharpe={sharpe:.4f} | Cumulative={cumulative*100:.2f}% | "
        f"N_trades={n_trades} | Win rate={win_rate*100:.1f}%"
    )
    return result
