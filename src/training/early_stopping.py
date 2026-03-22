"""
src/training/early_stopping.py
================================
EarlyStopping callback with best-weight restoration.

Monitors val_binary_accuracy by default (CONFIG['early_stopping_monitor']).
Stops training when no improvement for CONFIG['early_stopping_patience'] (35)
consecutive epochs. Restores best model weights when triggered.

Why val_binary_accuracy instead of val_loss for MTL:
  The uncertainty-weighted MTL loss (Kendall et al. CVPR 2018) combines MSE
  and BCE via two trainable σ parameters (log_sigma1, log_sigma2) that require
  approximately 20-30 epochs to stabilize. During σ convergence, the composite
  val_loss fluctuates non-monotonically even when the classification head is
  genuinely improving. Stopping on val_loss risks terminating training before
  σ parameters have converged, leaving the loss weighting suboptimal.
  val_binary_accuracy gives a clean, direct signal for the primary classification
  task and is unaffected by σ oscillations.

  ReduceLROnPlateau continues to monitor val_loss separately — that is correct
  behavior for composite optimization (LR schedule responds to overall loss
  stalling, not just the classification head).

  Reference: LSTM-Forest MTL paper (ScienceDirect 2021).
"""

import copy
import torch
import torch.nn as nn
from config.config import CONFIG


class EarlyStopping:
    """Monitor a metric and stop training when patience is exhausted.

    By default monitors val_binary_accuracy (CONFIG['early_stopping_monitor']).
    For accuracy metrics, higher is better; for loss metrics, lower is better —
    direction is inferred automatically from the monitor name.

    Why val_binary_accuracy for MTL:
      Uncertainty-weighted composite val_loss fluctuates non-monotonically
      during σ parameter convergence (first 20-30 epochs). val_binary_accuracy
      gives a cleaner signal for the primary classification task.
      LR scheduler monitors val_loss separately — separation of concerns.

    Args:
        monitor:      Metric name to monitor. Defaults to
                      CONFIG['early_stopping_monitor'] ('val_binary_accuracy').
                      If 'accuracy' appears in the name, higher is better;
                      otherwise lower is better (loss).
        patience:     Epochs to wait without improvement before stopping.
                      Defaults to CONFIG['early_stopping_patience'] (35).
        restore_best: Whether to restore best weights when stopped.
                      Defaults to CONFIG['restore_best_weights'] (True).
        min_delta:    Minimum change to count as improvement.

    Example:
        >>> es = EarlyStopping()
        >>> es(metric_value=0.56, model=model)
        >>> es.stop
        False
    """

    def __init__(self,
                 monitor: str = None,
                 patience: int = None,
                 restore_best: bool = None,
                 min_delta: float = 1e-6):
        self.monitor = monitor if monitor is not None \
            else CONFIG['early_stopping_monitor']
        self.patience = patience if patience is not None \
            else CONFIG['early_stopping_patience']
        self.restore_best = restore_best if restore_best is not None \
            else CONFIG['restore_best_weights']
        self.min_delta = min_delta
        # Accuracy → higher is better; loss → lower is better
        self._higher_is_better = 'accuracy' in self.monitor
        self.best_score: float = float('-inf') if self._higher_is_better \
            else float('inf')
        self.counter: int = 0
        self.stop: bool = False
        self._best_state: dict = None

    def __call__(self, metric_value: float, model: nn.Module) -> None:
        """Update state given current metric value.

        Args:
            metric_value: Current epoch metric (accuracy or loss value).
            model:        Model whose state_dict to checkpoint.

        Returns:
            None. Sets self.stop = True when patience exhausted.

        Example:
            >>> es(0.56, model)
            >>> es.counter
            0
        """
        if self._higher_is_better:
            improved = metric_value > self.best_score + self.min_delta
        else:
            improved = metric_value < self.best_score - self.min_delta

        if improved:
            self.best_score = metric_value
            self.counter = 0
            if self.restore_best:
                self._best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore(self, model: nn.Module) -> None:
        """Restore the best saved weights into model.

        Args:
            model: Model to load best weights into.

        Returns:
            None.

        Example:
            >>> es.restore(model)
        """
        if self._best_state is not None:
            model.load_state_dict(self._best_state)

    def reset(self) -> None:
        """Reset all state for a new training run.

        Args:
            None.

        Returns:
            None.

        Example:
            >>> es.reset()
            >>> es.counter
            0
        """
        self.best_score = float('-inf') if self._higher_is_better \
            else float('inf')
        self.counter = 0
        self.stop = False
        self._best_state = None
