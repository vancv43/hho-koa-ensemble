"""
Harris Hawks Optimization for ensemble soft-voting weight search.
Manuscript Algorithm 1 / Section 3.5, Eq. 7.

Heidari, A. A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M., &
Chen, H. (2019). Harris hawks optimization: Algorithm and applications.
Future Generation Computer Systems, 97, 849-872.

As stated in the manuscript (Section 3.5), this variant uses exploration
plus soft-besiege exploitation only; the hard-besiege and Levy-flight
branches of canonical HHO are omitted. Hawks are drawn i.i.d. from
U(0,1)^4 and mapped onto the simplex by softmax; E0 ~ U(-1,1) is redrawn
for every hawk at every iteration; all T iterations always run.
"""
import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def hho_optimize(fitness_fn, dim: int, population: int = 30, iterations: int = 80,
                  seed: int = 42):
    """Maximizes fitness_fn(weights) over the probability simplex (weights
    are produced by softmax so they always sum to 1 and stay non-negative).

    fitness_fn: callable(np.ndarray of shape (dim,)) -> float (e.g. validation QWK)
    Returns: (best_weights, best_fitness, history) where history is the
    best-fitness-per-iteration trace used for the convergence comparison
    (manuscript Tables 6-7, Figure 4).
    """
    rng = np.random.default_rng(seed)
    hawks = rng.uniform(0, 1, size=(population, dim))
    hawks = _softmax(hawks)

    fitness = np.array([fitness_fn(h) for h in hawks])
    best_idx = fitness.argmax()
    rabbit = hawks[best_idx].copy()
    best_fit = fitness[best_idx]
    history = [best_fit]

    for t in range(iterations):
        e0 = rng.uniform(-1, 1, size=population)
        e = 2 * e0 * (1 - t / iterations)

        for i in range(population):
            if abs(e[i]) >= 1:
                # exploration: random repositioning
                rand_hawk = hawks[rng.integers(population)]
                hawks[i] = rand_hawk - rng.random() * abs(rand_hawk - 2 * rng.random() * hawks[i])
            else:
                # exploitation: soft besiege toward the rabbit (best solution)
                jump_strength = 2 * (1 - rng.random())
                hawks[i] = (rabbit - hawks[i]) - e[i] * abs(jump_strength * rabbit - hawks[i])

            hawks[i] = np.clip(hawks[i], 0, 1)

        hawks = _softmax(hawks)
        fitness = np.array([fitness_fn(h) for h in hawks])
        gen_best = fitness.argmax()
        if fitness[gen_best] > best_fit:
            best_fit = fitness[gen_best]
            rabbit = hawks[gen_best].copy()
        history.append(best_fit)

    return rabbit, best_fit, history
