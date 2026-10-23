"""Ordinal evaluation metrics used throughout the pipeline."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> float:
    """QWK (manuscript Eq. 8), thin wrapper around sklearn's quadratic-weighted
    Cohen's kappa for readability / to keep the paper's own symbol names close
    to the code."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def ensemble_predict(probs_per_model: list, weights: np.ndarray) -> np.ndarray:
    """Soft-vote: argmax_c sum_i w_i * P_i(c) (manuscript Eq. 7)."""
    stacked = np.stack(probs_per_model, axis=0)  # (M, N, K)
    weighted = np.tensordot(weights, stacked, axes=(0, 0))  # (N, K)
    return weighted.argmax(axis=1)


def full_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict:
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_auc": roc_auc_score(y_true, probs / probs.sum(axis=1, keepdims=True),
                                   multi_class="ovr", average="macro"),
    }
