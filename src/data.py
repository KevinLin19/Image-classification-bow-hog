from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg"}

def load_class_dir(class_dir: str | Path, label: int) -> Tuple[List[np.ndarray], np.ndarray]:
    class_dir = Path(class_dir)
    images: List[np.ndarray] = []
    y: List[int] = []

    for p in sorted(class_dir.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            img = Image.open(p).convert("L")
            images.append(np.array(img, dtype=np.uint8))
            y.append(label)

    return images, np.array(y, dtype=int)