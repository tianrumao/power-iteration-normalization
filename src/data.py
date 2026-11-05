from dataclasses import dataclass

import numpy as np


@dataclass
class Samples:
    tokens: np.ndarray
    target: np.ndarray
    power: np.ndarray
    gap: np.ndarray


def sample(size, d, n, rng):
    x = rng.standard_normal((size, d, n))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    a = rng.standard_normal((size, d))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    s = x @ x.swapaxes(-1, -2)
    eigenvalues, eigenvectors = np.linalg.eigh(s)
    tokens = np.zeros((size, 2 * d, n + 1))
    tokens[:, :d, :n] = x
    tokens[:, d:, n] = a
    power = [a]
    for _ in range(4):
        update = (s @ power[-1][..., None])[..., 0]
        power.append(update / np.linalg.norm(update, axis=1, keepdims=True))
    return Samples(tokens, eigenvectors[:, :, -1], np.stack(power, axis=1),
                   eigenvalues[:, -1] - eigenvalues[:, -2])


def splits(config):
    seeds = np.random.SeedSequence(config["seed"]).spawn(3)
    return {
        name: sample(config[f"{name}_size"], config["d"], config["n"],
                     np.random.default_rng(seed))
        for name, seed in zip(("train", "validation", "test"), seeds)
    }
