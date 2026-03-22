"""
src/training/hyperparam_tuning.py
===================================
Random hyperparameter search over CONFIG['search_spaces'].

Done ONCE on Kotekar val set per model.
Best params saved and updated in config/config.py manually.

Responsibilities:
- evaluate_params() : train one trial and return val_loss
- random_search()   : sample 40 random configs, return best
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List
from config.config import CONFIG
from src.models.mtl_model import build_model
from src.models.losses import uncertainty_weighted_loss, fixed_weighted_loss
from src.training.early_stopping import EarlyStopping


def _sample_params(model_name: str) -> Dict[str, Any]:
    """Sample a random hyperparameter configuration for the given model.

    Draws from CONFIG['search_spaces']['common'] + model-specific space.

    Args:
        model_name: One of 'tkan', 'lstm', 'gru', 'tcn'.

    Returns:
        Dict of sampled hyperparameter values.

    Example:
        >>> params = _sample_params('lstm')
        >>> 'hidden_size' in params
        True
    """
    spaces = CONFIG['search_spaces']
    params = {k: random.choice(v) for k, v in spaces['common'].items()}
    if model_name in spaces:
        for k, v in spaces[model_name].items():
            params[k] = random.choice(v)
    return params


def evaluate_params(
    model_name: str,
    params: Dict[str, Any],
    n_features: int,
    data: Dict[str, np.ndarray],
    seed: int = 42,
    device: str = 'cpu',
) -> float:
    """Train one trial with given params and return val_loss.

    Uses a short training run (max 50 epochs) for search efficiency.

    Args:
        model_name: One of 'tkan', 'lstm', 'gru', 'tcn'.
        params:     Hyperparameter dict from _sample_params().
        n_features: Input feature dimension.
        data:       Dict with X_train, y_clf_train, y_reg_train, X_val, etc.
        seed:       RNG seed for this trial.
        device:     'cpu' or 'cuda'.

    Returns:
        Scalar val_loss (float). Lower is better.

    Example:
        >>> loss = evaluate_params('lstm', params, 7, data)
        >>> isinstance(loss, float)
        True
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Temporarily patch CONFIG to use trial params
    _orig = CONFIG['best_params'][model_name].copy()
    CONFIG['best_params'][model_name].update(params)

    try:
        model = build_model(model_name, CONFIG, n_features).to(device)
    finally:
        CONFIG['best_params'][model_name] = _orig

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
    )
    # Explicitly monitor val_loss during search trials (not val_binary_accuracy)
    # — hyperparameter selection optimises composite loss, not just accuracy.
    early_stop = EarlyStopping(patience=10, monitor='val_loss')
    bce_fn = nn.BCELoss()
    mse_fn = nn.MSELoss()
    loss_type = CONFIG['loss_type']

    def _tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    X_t = _tensor(data['X_train'])
    yc_t = _tensor(data['y_clf_train']).unsqueeze(1)
    yr_t = _tensor(data['y_reg_train']).unsqueeze(1)
    X_v = _tensor(data['X_val']).to(device)
    yc_v = _tensor(data['y_clf_val']).unsqueeze(1).to(device)
    yr_v = _tensor(data['y_reg_val']).unsqueeze(1).to(device)

    ds = TensorDataset(X_t, yc_t, yr_t)
    loader = DataLoader(ds, batch_size=params['batch_size'], shuffle=False)

    best_val_loss = float('inf')
    for _ in range(50):
        model.train()
        for X_b, yc_b, yr_b in loader:
            X_b, yc_b, yr_b = X_b.to(device), yc_b.to(device), yr_b.to(device)
            optimizer.zero_grad()
            reg_out, clf_out = model(X_b)
            mse = mse_fn(reg_out, yr_b)
            bce = bce_fn(clf_out, yc_b)
            if loss_type == 'uncertainty':
                loss = uncertainty_weighted_loss(
                    mse, bce, model.log_sigma1, model.log_sigma2)
            else:
                loss = fixed_weighted_loss(mse, bce)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            reg_v, clf_v = model(X_v)
            val_mse = mse_fn(reg_v, yr_v)
            val_bce = bce_fn(clf_v, yc_v)
            if loss_type == 'uncertainty':
                val_loss = uncertainty_weighted_loss(
                    val_mse, val_bce,
                    model.log_sigma1, model.log_sigma2).item()
            else:
                val_loss = fixed_weighted_loss(val_mse, val_bce).item()

        best_val_loss = min(best_val_loss, val_loss)
        early_stop(val_loss, model)
        if early_stop.stop:
            break

    return best_val_loss


def random_search(
    model_name: str,
    n_features: int,
    data: Dict[str, np.ndarray],
    n_trials: int = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Run random hyperparameter search and return best params.

    Args:
        model_name: One of 'tkan', 'lstm', 'gru', 'tcn'.
        n_features: Input feature dimension.
        data:       Dict with all train/val arrays.
        n_trials:   Number of random trials. Defaults to CONFIG['n_search_trials'] (40).
        device:     'cpu' or 'cuda'.

    Returns:
        Dict of best hyperparameter values (to be manually saved to config.py).

    Example:
        >>> best = random_search('lstm', 7, data)
        >>> 'hidden_size' in best
        True
    """
    if n_trials is None:
        n_trials = CONFIG['n_search_trials']

    best_loss = float('inf')
    best_params = None

    for trial in range(n_trials):
        params = _sample_params(model_name)
        val_loss = evaluate_params(
            model_name=model_name,
            params=params,
            n_features=n_features,
            data=data,
            seed=trial,
            device=device,
        )
        print(
            f"[{model_name}] Trial {trial + 1}/{n_trials} | "
            f"val_loss={val_loss:.6f} | params={params}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params

    print(f"\n[{model_name}] Best val_loss={best_loss:.6f} | Best params={best_params}")
    return best_params
