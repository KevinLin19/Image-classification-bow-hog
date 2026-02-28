from __future__ import annotations
import numpy as np

def generate_grid(width: int, height: int, n_points_x: int, n_points_y: int, margin: int = 8):
    """
    Returns x_indices, y_indices of grid points, avoiding borders by 'margin'
    so 16x16 patch centered at point stays inside image.
    """
    x_values = np.linspace(margin, width - 1 - margin, n_points_x, dtype=int)
    y_values = np.linspace(margin, height - 1 - margin, n_points_y, dtype=int)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    return x_grid.ravel(), y_grid.ravel()