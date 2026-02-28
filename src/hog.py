from __future__ import annotations
import numpy as np
from scipy.ndimage import sobel

def compute_hog_descriptors(image: np.ndarray, x_points: np.ndarray, y_points: np.ndarray,
                            cell_size: int = 4, num_bins: int = 8) -> np.ndarray:
    """
    Returns (n_points, 128) float32 descriptors.
    HOG on 16x16 patch split into 4x4 cells, 8 bins per cell.
    Unsigned orientations in [0, pi).
    """
    img = image.astype(np.float32)
    n_points = len(x_points)
    out = np.zeros((n_points, 4 * 4 * num_bins), dtype=np.float32)

    for i, (x, y) in enumerate(zip(x_points, y_points)):
        patch = img[y-8:y+8, x-8:x+8]  # 16x16
        if patch.shape != (16, 16):
            continue

        gx = sobel(patch, axis=1)
        gy = sobel(patch, axis=0)
        mag = np.hypot(gx, gy)

        # angle in [0, pi)
        ang = (np.arctan2(gy, gx) + np.pi) % np.pi

        blocks = []
        for cy in range(0, 16, cell_size):
            for cx in range(0, 16, cell_size):
                cell_mag = mag[cy:cy+cell_size, cx:cx+cell_size]
                cell_ang = ang[cy:cy+cell_size, cx:cx+cell_size]
                hist, _ = np.histogram(cell_ang, bins=num_bins, range=(0.0, np.pi), weights=cell_mag)
                blocks.append(hist.astype(np.float32))

        out[i] = np.concatenate(blocks)

    return out