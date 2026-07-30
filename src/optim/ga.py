"""
Genetic Algorithm baseline for ensemble weight search
(manuscript Section 4.4, Table 6 comparison).

Holland, J. H. (1992). Genetic algorithms. Scientific American, 267(1), 66-72.
"""
import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _tournament_select(pop: np.ndarray, fitness: np.ndarray, rng, k: int = 3) -> np.ndarray:
    idx = rng.integers(0, len(pop), size=k)
    winner = idx[fitness[idx].argmax()]
    return pop[winner]


def ga_optimize(fitness_fn, dim: int, population: int = 30, iterations: int = 80,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    pop = _softmax(rng.uniform(0, 1, size=(population, dim)))
    fitness = np.array([fitness_fn(p) for p in pop])
    best_idx = fitness.argmax()
    best, best_fit = pop[best_idx].copy(), fitness[best_idx]
    history = [best_fit]

    for _ in range(iterations):
        new_pop = []
        while len(new_pop) < population:
            parent1 = _tournament_select(pop, fitness, rng)
            parent2 = _tournament_select(pop, fitness, rng)
            if rng.random() < crossover_rate:
                alpha = rng.random()
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                child = parent1.copy()
            if rng.random() < mutation_rate:
                child = child + rng.normal(0, 0.1, size=dim)
            new_pop.append(child)

        pop = _softmax(np.stack(new_pop))
        fitness = np.array([fitness_fn(p) for p in pop])
        gen_best = fitness.argmax()
        if fitness[gen_best] > best_fit:
            best_fit = fitness[gen_best]
            best = pop[gen_best].copy()
        history.append(best_fit)

    return best, best_fit, history
