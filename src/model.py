# src/model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from .data import load_class_dir
from .grid import generate_grid
from .hog import compute_hog_descriptors
from .kmeans import kmeans
from .bow import bow_histograms
from .knn import knn_predict


@dataclass
class BoWModel:
    centroids: np.ndarray           # (K, 128)
    train_bow: np.ndarray           # (T, K)
    train_labels: np.ndarray        # (T,)
    width: int
    height: int
    n_points_x: int
    n_points_y: int
    metric: str
    class_names: list[str]          # index -> class name

    def predict_image(self, image_path: str, resize_if_needed: bool = True) -> int:
        img = Image.open(image_path).convert("L")
        if resize_if_needed:
            img = img.resize((self.width, self.height))

        arr = np.array(img, dtype=np.uint8)
        xg, yg = generate_grid(self.width, self.height, self.n_points_x, self.n_points_y, margin=8)
        desc = compute_hog_descriptors(arr, xg, yg)
        bow = bow_histograms([desc], self.centroids)  # (1, K)
        pred = knn_predict(self.train_bow, self.train_labels, bow, metric=self.metric)
        return int(pred[0])

    def predict_label(self, image_path: str) -> str:
        idx = self.predict_image(image_path)
        return self.class_names[idx]

    def predict_many(self, paths: list[str]) -> list[tuple[str, str]]:
        """
        Predict each image path independently.
        Returns list of (image_path, predicted_label).
        """
        results = []
        for p in paths:
            label = self.predict_label(p)
            results.append((p, label))
        return results

    def predict_folder(self, folder: str) -> list[tuple[str, str]]:
        """
        Predict all images inside a folder.
        """
        p = Path(folder)
        exts = (".png", ".jpg", ".jpeg")
        images = sorted([x for x in p.iterdir() if x.suffix.lower() in exts])
        return self.predict_many([str(x) for x in images])

    def predict_with_vote(self, paths: list[str]) -> tuple[str, dict[str, int]]:
        """
        Majority vote across multiple images.
        Returns (final_label, counts).
        """
        preds = [self.predict_label(p) for p in paths]
        counts: dict[str, int] = {}
        for lab in preds:
            counts[lab] = counts.get(lab, 0) + 1
        final = max(counts.items(), key=lambda kv: kv[1])[0]
        return final, counts


def train_multiclass_model(
    train_root: str,
    class_names: list[str],
    K: int = 200,
    metric: str = "l2",
    n_points_x: int = 8,
    n_points_y: int = 8,
    seed: int = 0,
) -> BoWModel:
    train_root = Path(train_root)

    X_train: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for label, cname in enumerate(class_names):
        imgs, y = load_class_dir(train_root / cname, label=label)
        X_train.extend(imgs)
        y_list.append(y)

    y_train = np.concatenate(y_list)

    h, w = X_train[0].shape
    xg, yg = generate_grid(w, h, n_points_x, n_points_y, margin=8)

    train_desc = [compute_hog_descriptors(img, xg, yg)
                  for img in tqdm(X_train, desc="HOG train (model save)")
                  ]
    X_all = np.vstack(train_desc)

    centroids, _ = kmeans(X_all, K=K, max_iters=10, seed=seed)
    train_bow = bow_histograms(train_desc, centroids)

    return BoWModel(
        centroids=centroids,
        train_bow=train_bow,
        train_labels=y_train,
        width=w,
        height=h,
        n_points_x=n_points_x,
        n_points_y=n_points_y,
        metric=metric,
        class_names=class_names,
    )


def save_model(model: BoWModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        centroids=model.centroids,
        train_bow=model.train_bow,
        train_labels=model.train_labels,
        width=model.width,
        height=model.height,
        n_points_x=model.n_points_x,
        n_points_y=model.n_points_y,
        metric=model.metric,
        class_names=np.array(model.class_names, dtype=object),
    )


def load_model(path: str | Path) -> BoWModel:
    d = np.load(path, allow_pickle=True)
    return BoWModel(
        centroids=d["centroids"],
        train_bow=d["train_bow"],
        train_labels=d["train_labels"],
        width=int(d["width"]),
        height=int(d["height"]),
        n_points_x=int(d["n_points_x"]),
        n_points_y=int(d["n_points_y"]),
        metric=str(d["metric"]),
        class_names=list(d["class_names"]),
    )