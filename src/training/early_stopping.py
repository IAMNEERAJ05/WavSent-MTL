"""
src/training/early_stopping.py
================================
EarlyStopping callback with best-weight restoration.

Monitors validation loss. Stops training when no improvement
for CONFIG['early_stopping_patience'] (15) consecutive epochs.
Restores best model weights when triggered.
"""

import copy
import torch
import torch.nn as nn
from config.config import CONFIG


class EarlyStopping:
    """Monitor validation loss and stop training when patience is exhausted.

    Restores best model weights upon stopping (CONFIG['restore_best_weights']=True).

    Args:
        patience:       Epochs to wait without improvement before stopping.
                        Defaults to CONFIG['early_stopping_patience'] (15).
        restore_best:   Whether to restore best weights when stopped.
                        Defaults to CONFIG['restore_best_weights'] (True).
        min_delta:      Minimum change to count as improvement.

    Example:
        >>> es = EarlyStopping(patience=15)
        >>> es(val_loss=0.5, model=model)
        >>> es.stop
        False
    """

    def __init__(self,
                 patience: int = None,
                 restore_best: bool = None,
                 min_delta: float = 1e-6):
        self.patience = patience if patience is not None \
            else CONFIG['early_stopping_patience']
        self.restore_best = restore_best if restore_best is not None \
            else CONFIG['restore_best_weights']
        self.min_delta = min_delta
        self.best_loss: float = float('inf')
        self.counter: int = 0
        self.stop: bool = False
        self._best_state: dict = None

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """Update state given current validation loss.

        Args:
            val_loss: Current epoch validation loss.
            model:    Model whose state_dict to checkpoint.

        Returns:
            None. Sets self.stop = True when patience exhausted.

        Example:
            >>> es(0.45, model)
            >>> es.counter
            0
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
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
        self.best_loss = float('inf')
        self.counter = 0
        self.stop = False
        self._best_state = None
