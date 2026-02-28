# src/experiments.py
from __future__ import annotations

from pathlib import Path
import numpy as np
from tqdm import tqdm

from .data import load_class_dir
from .grid import generate_grid
from .hog import compute_hog_descriptors
from .kmeans import kmeans
from .bow import bow_histograms
from .knn import knn_predict


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def run_multiclasses(
    train_root: str,
    test_root: str,
    class_names: list[str],
    K: int = 100,
    runs: int = 10,
    metric: str = "l2",
    n_points_x: int = 8,
    n_points_y: int = 8,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Generic experiment runner for N classes (works for 2 classes too).

    Ex:
      run_multiclasses("data/stl10_raw/train", "data/stl10_raw/test", ["bird","airplane"], K=50, runs=1)
      run_multiclasses("data/stl10_raw/train", "data/stl10_raw/test",
                       ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"])
    """

    train_root = Path(train_root)
    test_root = Path(test_root)

    X_train: list[np.ndarray] = []
    y_train_list: list[np.ndarray] = []
    X_test: list[np.ndarray] = []
    y_test_list: list[np.ndarray] = []

    # Load data for each class and assign label 0..C-1
    for label, cname in enumerate(class_names):
        imgs_tr, y_tr = load_class_dir(train_root / cname, label=label)
        imgs_te, y_te = load_class_dir(test_root / cname, label=label)

        X_train.extend(imgs_tr)
        y_train_list.append(y_tr)

        X_test.extend(imgs_te)
        y_test_list.append(y_te)

    y_train = np.concatenate(y_train_list)
    y_test = np.concatenate(y_test_list)

    # Grid based on image size (assume all same size)
    h, w = X_train[0].shape
    xg, yg = generate_grid(width=w, height=h, n_points_x=n_points_x, n_points_y=n_points_y, margin=8)

    accs: list[float] = []
    for r in range(runs):
        train_desc = [compute_hog_descriptors(img, xg, yg)
                      for img in tqdm(X_train, desc=f"[Run {r+1}/{runs}] HOG train", leave=False)
                      ]
        test_desc = [compute_hog_descriptors(img, xg, yg)
                     for img in tqdm(X_test, desc=f"[Run {r+1}/{runs}] HOG test", leave=False)
                     ]

        X_all = np.vstack(train_desc)
        print(f"[Run {r+1}/{runs}] K-means on {X_all.shape[0]} descriptors, K={K}...")
        centroids, _ = kmeans(X_all, K=K, max_iters=10, seed=seed + r)

        train_bow = bow_histograms(train_desc, centroids)
        test_bow = bow_histograms(test_desc, centroids)

        pred = knn_predict(train_bow, y_train, test_bow, metric=metric)
        accs.append(accuracy(y_test, pred))

    return float(np.mean(accs)), float(np.std(accs))