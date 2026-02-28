from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist

def kmeans(X: np.ndarray, K: int, max_iters: int = 10, tol: float = 1e-6, seed: int | None = None):
    rng = np.random.default_rng(seed)
    n_samples, _ = X.shape
    init_idx = rng.permutation(n_samples)[:K]
    centroids = X[init_idx].copy()

    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iters):
        d = cdist(X, centroids, metric="euclidean")
        new_labels = np.argmin(d, axis=1)

        new_centroids = centroids.copy()
        for k in range(K):
            mask = (new_labels == k)
            if np.any(mask):
                new_centroids[k] = X[mask].mean(axis=0)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        labels = new_labels

        if shift < tol:
            break

    return centroids, labels