from __future__ import annotations
import numpy as np

def knn_predict(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, metric: str = "l2") -> np.ndarray:
    pred = np.zeros(len(test_X), dtype=int)

    for i, x in enumerate(test_X):
        if metric == "l2":
            d = np.linalg.norm(train_X - x, axis=1)
        elif metric == "l1":
            d = np.sum(np.abs(train_X - x), axis=1)
        else:
            raise ValueError("metric must be 'l1' or 'l2'")
        pred[i] = train_y[np.argmin(d)]

    return pred