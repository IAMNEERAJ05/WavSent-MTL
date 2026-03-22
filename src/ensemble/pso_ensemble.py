"""
src/ensemble/pso_ensemble.py
==============================
PSO-based ensemble weight search over saved validation predictions.

CRITICAL: PSO searches on VALIDATION predictions ONLY — not training.
          Individual model test metrics must be stored BEFORE ensemble evaluation.

Protocol per DECISIONS.md:
  n_particles=20, iterations=50, c1=0.5, c2=0.3, w=0.9
  Weights softmax-normalized → [w1,w2,w3,w4]
  Fitness = negative val accuracy (PSO minimizes)

Responsibilities:
- collect_val_predictions()   : load saved .npy val/test probs per model
- run_pso_search()            : search optimal weights on val predictions
- apply_ensemble_weights()    : apply learned weights to test predictions
"""

import os
import numpy as np
import pyswarms as ps
from typing import Dict, List, Tuple
from config.config import CONFIG
from src.evaluation.metrics import compute_clf_metrics


def _softmax(w: np.ndarray) -> np.ndarray:
    """Apply softmax normalization to weight vector.

    Args:
        w: 1-D array of raw weights.

    Returns:
        Softmax-normalized 1-D array summing to 1.

    Example:
        >>> _softmax(np.array([1.0, 2.0, 3.0, 4.0])).sum()
        1.0
    """
    e = np.exp(w - np.max(w))
    return e / e.sum()


def collect_val_predictions(
    dataset: str,
    model_names: List[str] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load saved val and test prediction arrays for each model.

    Expects files at:
      ablation/results/{dataset}/val_predictions/{model}_val_preds.npy
      ablation/results/{dataset}/val_predictions/{model}_test_preds.npy

    Args:
        dataset:     'kotekar' or 'kaggle'.
        model_names: List of model names. Defaults to CONFIG['pso_models'].

    Returns:
        Tuple (val_preds_dict, test_preds_dict):
        - val_preds_dict:  {model_name: [n_val,] float32 array}
        - test_preds_dict: {model_name: [n_test,] float32 array}

    Example:
        >>> val_p, test_p = collect_val_predictions('kotekar')
        >>> 'tkan' in val_p
        True
    """
    if model_names is None:
        model_names = CONFIG['pso_models']

    pred_dir = os.path.join(CONFIG['ablation_dir'], dataset, 'val_predictions')
    val_preds = {}
    test_preds = {}

    for m in model_names:
        val_path = os.path.join(pred_dir, f'{m}_val_preds.npy')
        test_path = os.path.join(pred_dir, f'{m}_test_preds.npy')
        if not os.path.exists(val_path):
            raise FileNotFoundError(
                f"Val predictions not found: {val_path}\n"
                f"Run training first to generate predictions."
            )
        val_preds[m] = np.load(val_path)
        test_preds[m] = np.load(test_path)

    return val_preds, test_preds


def run_pso_search(
    val_preds: Dict[str, np.ndarray],
    val_labels: np.ndarray,
    model_names: List[str] = None,
) -> Dict[str, float]:
    """Search for optimal ensemble weights using PSO on validation predictions.

    Uses pyswarms GlobalBestPSO. Fitness = negative val accuracy.
    Weights are softmax-normalized so they sum to 1.

    Args:
        val_preds:   {model_name: [n_val,] float32} from collect_val_predictions().
        val_labels:  [n_val,] int binary ground-truth labels.
        model_names: Ordered list of model names (must match val_preds keys).
                     Defaults to CONFIG['pso_models'].

    Returns:
        Dict {model_name: weight_float} where weights sum to 1.0.

    Example:
        >>> weights = run_pso_search(val_preds, y_clf_val)
        >>> abs(sum(weights.values()) - 1.0) < 1e-6
        True
    """
    if model_names is None:
        model_names = CONFIG['pso_models']

    n_models = len(model_names)
    preds_matrix = np.stack(
        [val_preds[m] for m in model_names], axis=0
    )  # [n_models, n_val]

    def fitness(particles: np.ndarray) -> np.ndarray:
        """PSO fitness function — negative accuracy (minimize).

        Args:
            particles: [n_particles, n_models] raw weight matrix.

        Returns:
            [n_particles,] negative accuracy for each particle.
        """
        costs = []
        for particle in particles:
            w = _softmax(particle)
            ensemble_probs = (w[:, None] * preds_matrix).sum(axis=0)
            acc = compute_clf_metrics(val_labels, ensemble_probs)['accuracy']
            costs.append(-acc)  # minimize negative accuracy
        return np.array(costs)

    options = {
        'c1': CONFIG['pso_c1'],
        'c2': CONFIG['pso_c2'],
        'w':  CONFIG['pso_w'],
    }
    optimizer = ps.single.GlobalBestPSO(
        n_particles=CONFIG['pso_n_particles'],
        dimensions=n_models,
        options=options,
    )
    best_cost, best_pos = optimizer.optimize(
        fitness,
        iters=CONFIG['pso_iterations'],
        verbose=True,
    )
    best_weights = _softmax(best_pos)
    MIN_WEIGHT = 0.10  # each model gets at least 10%
    best_weights = np.clip(best_weights, MIN_WEIGHT, None)
    best_weights = best_weights / best_weights.sum()  # renormalize to sum=1
    weights_dict = {m: float(best_weights[i]) for i, m in enumerate(model_names)}

    print(f"PSO best val accuracy: {-best_cost:.4f}")
    print(f"PSO weights: {weights_dict}")
    return weights_dict


def apply_ensemble_weights(
    weights: Dict[str, float],
    test_preds: Dict[str, np.ndarray],
    test_labels: np.ndarray,
    model_names: List[str] = None,
) -> Dict[str, float]:
    """Apply PSO-learned weights to test predictions and compute metrics.

    CRITICAL: Individual model test metrics should already be stored
              before calling this function.

    Args:
        weights:      {model_name: float} from run_pso_search().
        test_preds:   {model_name: [n_test,] float32} test probabilities.
        test_labels:  [n_test,] int binary ground-truth labels.
        model_names:  Ordered list. Defaults to CONFIG['pso_models'].

    Returns:
        Dict of Config G test classification metrics.

    Example:
        >>> g_metrics = apply_ensemble_weights(weights, test_preds, y_clf_test)
        >>> 'accuracy' in g_metrics
        True
    """
    if model_names is None:
        model_names = CONFIG['pso_models']

    ensemble_probs = np.zeros_like(test_preds[model_names[0]], dtype=np.float32)
    for m in model_names:
        ensemble_probs += weights[m] * test_preds[m]

    metrics = compute_clf_metrics(test_labels, ensemble_probs)
    print(f"Config G (ensemble) test metrics: {metrics}")
    return metrics
