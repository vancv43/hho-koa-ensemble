"""
Reproduce the manuscript results that depend only on the cached TTA-8
backbone probabilities in data/tta8_probs.npz (no images, no GPU needed):

  Table 7   optimizer comparison over 30 independent runs (seeds 0-29)
  Table 8   TTA rows: Unif-Ens+TTA and HHO-Ens+TTA
  Table 9   paired bootstrap validation (B=10,000, seed 42)
  Table 10  per-class metrics of HHO-Ensemble with TTA

The deployed HHO weights (Table 6, Figure 1) are read from
logs/13_tta8_ensemble_search_seed42_log.txt.

Usage:
    python reproduce_tables.py            # writes results/reproduced/*.csv
"""
import os

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import classification_report

from evaluate import bootstrap_qwk_gain
from src.metrics import full_metrics, quadratic_weighted_kappa
from src.optim import ga_optimize, hho_optimize, pso_optimize

BACKBONES = ["DenseNet-121", "EfficientNetV2-S", "ResNet-50", "ConvNeXt-Tiny"]
DEPLOYED_WEIGHTS = np.array([0.3071, 0.2307, 0.2377, 0.2244])  # HHO, seed 42
OUT_DIR = os.path.join("results", "reproduced")


def soft_vote(stack, weights):
    return np.tensordot(weights, stack, axes=(0, 0))


def converge_iter(history):
    """First iteration whose best-so-far fitness reaches 0.999 x final best."""
    return next(i for i, v in enumerate(history) if v >= 0.999 * history[-1])


def table7(val, y_val, test, y_test):
    fitness = lambda w: quadratic_weighted_kappa(y_val, soft_vote(val, w).argmax(axis=1))
    rows, val_qwk = [], {}
    for name, optimizer in [("HHO", hho_optimize), ("PSO", pso_optimize), ("GA", ga_optimize)]:
        v, t, it = [], [], []
        for seed in range(30):
            w, best, history = optimizer(fitness, 4, population=30, iterations=80, seed=seed)
            v.append(best)
            t.append(quadratic_weighted_kappa(y_test, soft_vote(test, w).argmax(axis=1)))
            it.append(converge_iter(history))
        val_qwk[name] = np.array(v)
        q1, med, q3 = np.percentile(it, [25, 50, 75])
        rows.append(f"{name},{np.mean(v):.4f},{np.std(v, ddof=1):.4f},{np.mean(t):.4f},"
                    f"{np.std(t, ddof=1):.4f},{med:g},{q1:g},{q3:g}")
    p_pso = wilcoxon(val_qwk["HHO"], val_qwk["PSO"]).pvalue
    p_ga = wilcoxon(val_qwk["HHO"], val_qwk["GA"]).pvalue
    header = "optimizer,val_qwk_mean,val_qwk_sd,test_qwk_mean,test_qwk_sd,median_iter,iter_q1,iter_q3"
    footer = f"# Wilcoxon on per-seed val QWK: HHO vs PSO p={p_pso:.4f}, HHO vs GA p={p_ga:.2f}"
    return "\n".join([header] + rows + [footer])


def main():
    d = np.load(os.path.join("data", "tta8_probs.npz"))
    val = np.stack([d[f"val_{b}"] for b in BACKBONES])
    test = np.stack([d[f"test_{b}"] for b in BACKBONES])
    y_val, y_test = d["val_y"], d["test_y"]
    uniform = np.full(4, 0.25)
    os.makedirs(OUT_DIR, exist_ok=True)

    t7 = table7(val, y_val, test, y_test)

    lines = ["configuration,accuracy_pct,balanced_accuracy_pct,qwk,macro_auc"]
    for name, w in [("Unif-Ens+TTA", uniform), ("HHO-Ens+TTA", DEPLOYED_WEIGHTS)]:
        p = soft_vote(test, w)
        m = full_metrics(y_test, p.argmax(axis=1), p)
        lines.append(f"{name},{100 * m['accuracy']:.1f},{100 * m['balanced_accuracy']:.1f},"
                     f"{m['qwk']:.4f},{m['macro_auc']:.4f}")
    t8 = "\n".join(lines)

    lines = ["comparison,point_delta_qwk,ci95_low,ci95_high,p_one_sided"]
    effnet_only = np.array([0.0, 1.0, 0.0, 0.0])
    for name, ref in [("HHO vs M1 (uniform ensemble)", uniform),
                      ("HHO vs M2 (EfficientNetV2-S+TTA)", effnet_only)]:
        b = bootstrap_qwk_gain(list(test), y_test, DEPLOYED_WEIGHTS, ref, n_boot=10000, seed=42)
        lines.append(f"{name},{b['point_delta_qwk']:+.4f},{b['ci_95'][0]:.4f},"
                     f"{b['ci_95'][1]:.4f},{b['p_one_sided']:.2f}")
    t9 = "\n".join(lines)

    t10 = classification_report(
        y_test, soft_vote(test, DEPLOYED_WEIGHTS).argmax(axis=1),
        target_names=["KL0 Normal", "KL1 Doubtful", "KL2 Mild", "KL3 Moderate", "KL4 Severe"],
        digits=2)

    for fname, text in [("table7_multiseed.csv", t7), ("table8_tta_rows.csv", t8),
                        ("table9_bootstrap.csv", t9), ("table10_per_class_tta.txt", t10)]:
        with open(os.path.join(OUT_DIR, fname), "w") as f:
            f.write(text + "\n")
        print(f"== {fname}\n{text}\n")


if __name__ == "__main__":
    main()
