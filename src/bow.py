from __future__ import annotations
import numpy as np

def bow_histograms(per_image_desc: list[np.ndarray], centroids: np.ndarray) -> np.ndarray:
    """
    per_image_desc: list of (n_points, 128) arrays
    returns: (n_images, K) histograms
    """
    K = centroids.shape[0]
    hists = np.zeros((len(per_image_desc), K), dtype=np.float32)

    for i, desc in enumerate(per_image_desc):
        # distances (n_points, K)
        d = np.linalg.norm(desc[:, None, :] - centroids[None, :, :], axis=2)
        nearest = np.argmin(d, axis=1)
        h = np.bincount(nearest, minlength=K).astype(np.float32)

        # optional: normalize to frequency (better for L1/L2 comparison)
        if h.sum() > 0:
            h /= h.sum()

        hists[i] = h

    return hists