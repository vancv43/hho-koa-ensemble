"""
Test-set evaluation: full metrics (QWK, BalAcc, Acc, macro-AUC) plus the
paired bootstrap significance test for the HHO-vs-uniform ensemble gain
(manuscript Section 4.6, Table 9).

Usage:
    python evaluate.py --probs-dir ./cached_test_probs --labels test_labels.npy --weights 0.307 0.231 0.238 0.224
"""
import argparse

import numpy as np

from src.metrics import ensemble_predict, full_metrics, quadratic_weighted_kappa


def bootstrap_qwk_gain(probs_per_model, y_test, weights_a, weights_b, n_boot: int = 10000, seed: int = 42):
    """Paired bootstrap over the test set: Delta QWK = QWK(weights_a) - QWK(weights_b).
    Returns point estimate, 95% CI, and the one-sided p-value against H0: Delta <= 0.
    """
    rng = np.random.default_rng(seed)
    stack = np.stack(probs_per_model, axis=0)  # (M, N, K)
    n = stack.shape[1]

    def qwk_for(idx, weights):
        sub = stack[:, idx, :]
        weighted = np.tensordot(weights, sub, axes=(0, 0))
        return quadratic_weighted_kappa(y_test[idx], weighted.argmax(axis=1))

    point = qwk_for(np.arange(n), weights_a) - qwk_for(np.arange(n), weights_b)

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = qwk_for(idx, weights_a) - qwk_for(idx, weights_b)

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    p_value = float((deltas <= 0).mean())
    return {"point_delta_qwk": point, "ci_95": (ci_low, ci_high), "p_one_sided": p_value}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True,
                         help="HHO-optimized ensemble weights, one per backbone")
    parser.add_argument("--backbones", nargs="+",
                         default=["densenet121", "efficientnetv2_s", "resnet50", "convnext_tiny"])
    args = parser.parse_args()

    probs_per_model = [np.load(f"{args.probs_dir}/{b}.npy") for b in args.backbones]
    y_test = np.load(args.labels)
    weights = np.array(args.weights)
    uniform = np.ones_like(weights) / len(weights)

    y_pred = ensemble_predict(probs_per_model, weights)
    probs_ensemble = np.tensordot(weights, np.stack(probs_per_model, axis=0), axes=(0, 0))
    print("HHO-ensemble metrics:", full_metrics(y_test, y_pred, probs_ensemble))

    boot = bootstrap_qwk_gain(probs_per_model, y_test, weights, uniform)
    print("Bootstrap HHO vs. uniform:", boot)
