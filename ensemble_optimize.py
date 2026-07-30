"""
Phase 2: search the soft-voting weight simplex on validation QWK using
HHO (benchmarked against PSO and GA). Manuscript Section 3.5, Table 6.

Usage:
    python ensemble_optimize.py --probs-dir ./cached_val_probs --labels val_labels.npy
"""
import argparse

import numpy as np

import config
from src.metrics import ensemble_predict, quadratic_weighted_kappa
from src.optim import ga_optimize, hho_optimize, pso_optimize


def make_fitness_fn(probs_per_model, y_val):
    def fitness(weights: np.ndarray) -> float:
        y_pred = ensemble_predict(probs_per_model, weights)
        return quadratic_weighted_kappa(y_val, y_pred)
    return fitness


def compare_optimizers(probs_per_model, y_val):
    fitness_fn = make_fitness_fn(probs_per_model, y_val)
    dim = len(probs_per_model)

    results = {}
    for name, optimizer in [("HHO", hho_optimize), ("PSO", pso_optimize), ("GA", ga_optimize)]:
        weights, best_fit, history = optimizer(
            fitness_fn, dim,
            population=config.ENSEMBLE_POPULATION,
            iterations=config.ENSEMBLE_ITERATIONS,
            seed=config.RANDOM_SEED,
        )
        converge_iter = next((i for i, v in enumerate(history) if v >= 0.999 * best_fit), len(history))
        results[name] = {"weights": weights.tolist(), "val_qwk": best_fit, "converge_iter": converge_iter}
        print(f"{name}: val QWK={best_fit:.4f}  weights={np.round(weights, 3)}  "
              f"converged at iter {converge_iter}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs-dir", required=True,
                         help="directory with one <backbone>.npy softmax-probability file per model")
    parser.add_argument("--labels", required=True, help=".npy file of validation labels")
    args = parser.parse_args()

    probs_per_model = [np.load(f"{args.probs_dir}/{b}.npy") for b in config.BACKBONES]
    y_val = np.load(args.labels)
    compare_optimizers(probs_per_model, y_val)
