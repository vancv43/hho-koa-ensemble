"""
HHO-based binary feature selection for the tabular clinical branch
(manuscript Section 3.6). Operates on a generic (X, y) tabular dataset;
the actual Clinical Data-2 cohort (160 patients, 67 features) is not
distributed with this repository -- it is confidential patient data
(see the manuscript's Ethical Statement). Point this at your own
feature matrix with the same shape to reproduce the wrapper search.
"""
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def hho_feature_select(X: np.ndarray, y: np.ndarray, population: int = 15,
                        iterations: int = 30, cv_folds: int = 5, seed: int = 42):
    """Binary HHO wrapper: each hawk is a bit-mask over the feature columns,
    scored by 5-fold linear-SVM accuracy. Returns the best mask found.
    """
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    hawks = (rng.random((population, n_features)) > 0.5).astype(float)

    def score(mask_real: np.ndarray) -> float:
        mask = mask_real > 0.5
        if mask.sum() == 0:
            return 0.0
        clf = SVC(kernel="linear")
        return cross_val_score(clf, X[:, mask], y, cv=cv_folds).mean()

    fitness = np.array([score(h) for h in hawks])
    best_idx = fitness.argmax()
    rabbit, best_fit = hawks[best_idx].copy(), fitness[best_idx]

    for t in range(iterations):
        e0 = rng.uniform(-1, 1, size=population)
        e = 2 * e0 * (1 - t / iterations)
        for i in range(population):
            if abs(e[i]) >= 1:
                rand_hawk = hawks[rng.integers(population)]
                hawks[i] = rand_hawk - rng.random() * abs(rand_hawk - 2 * rng.random() * hawks[i])
            else:
                jump_strength = 2 * (1 - rng.random())
                hawks[i] = (rabbit - hawks[i]) - e[i] * abs(jump_strength * rabbit - hawks[i])
        hawks = 1 / (1 + np.exp(-hawks))  # sigmoid transfer to [0,1] before re-binarizing in score()

        fitness = np.array([score(h) for h in hawks])
        gen_best = fitness.argmax()
        if fitness[gen_best] > best_fit:
            best_fit = fitness[gen_best]
            rabbit = hawks[gen_best].copy()

    return (rabbit > 0.5), best_fit


def evaluate_selected_features(X: np.ndarray, y: np.ndarray, mask: np.ndarray, cv_folds: int = 5):
    """Train/evaluate an RBF-SVM on the HHO-selected feature subset
    (manuscript Table 12)."""
    clf = SVC(kernel="rbf")
    scores = cross_val_score(clf, X[:, mask], y, cv=cv_folds)
    return {"acc_mean": scores.mean(), "acc_std": scores.std(), "n_features": int(mask.sum())}
