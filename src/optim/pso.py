"""
Particle Swarm Optimization baseline for ensemble weight search
(manuscript Sections 4.3-4.4, Tables 6-7).

Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
Proceedings of ICNN'95, 4, 1942-1948.
"""
import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def pso_optimize(fitness_fn, dim: int, population: int = 30, iterations: int = 80,
                  omega: float = 0.5, c1: float = 1.5, c2: float = 1.5, seed: int = 42):
    rng = np.random.default_rng(seed)
    pos = _softmax(rng.uniform(0, 1, size=(population, dim)))
    vel = np.zeros_like(pos)

    fitness = np.array([fitness_fn(p) for p in pos])
    pbest, pbest_fit = pos.copy(), fitness.copy()
    gbest_idx = fitness.argmax()
    gbest, gbest_fit = pos[gbest_idx].copy(), fitness[gbest_idx]
    history = [gbest_fit]

    for _ in range(iterations):
        r1, r2 = rng.random(pos.shape), rng.random(pos.shape)
        vel = omega * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = _softmax(pos + vel)

        fitness = np.array([fitness_fn(p) for p in pos])
        improved = fitness > pbest_fit
        pbest[improved] = pos[improved]
        pbest_fit[improved] = fitness[improved]

        gen_best = fitness.argmax()
        if fitness[gen_best] > gbest_fit:
            gbest_fit = fitness[gen_best]
            gbest = pos[gen_best].copy()
        history.append(gbest_fit)

    return gbest, gbest_fit, history
