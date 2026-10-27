import numpy as np


def normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def dot(a, b):
    return np.sum(a * b, axis=-1)


def reflect(d, n):
    return d - 2.0 * dot(d, n)[..., None] * n
