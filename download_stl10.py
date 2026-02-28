import os
import tarfile
import urllib.request
import numpy as np
from pathlib import Path
from PIL import Image

URL = "https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
DATA_DIR = Path("data")
EXTRACT_DIR = DATA_DIR / "stl10_binary"
OUTPUT_DIR = DATA_DIR / "stl10_raw"

CLASS_NAMES = [
    "airplane","bird","car","cat","deer",
    "dog","horse","monkey","ship","truck"
]


def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    archive_path = DATA_DIR / "stl10_binary.tar.gz"

    if archive_path.exists():
        print("Dataset archive already downloaded.")
        return archive_path

    print("Downloading STL-10 dataset...")
    print("This may take a while depending on your internet connection.")
    urllib.request.urlretrieve(URL, archive_path)
    print("Download complete.")
    return archive_path


def extract_dataset(archive_path):
    if EXTRACT_DIR.exists():
        print("Dataset already extracted.")
        return

    print("Extracting dataset...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print("Extraction complete.")


def read_images(bin_file):
    with open(bin_file, "rb") as f:
        data = np.fromfile(f, dtype=np.uint8)
    images = data.reshape(-1, 3, 96, 96)
    images = images.transpose(0, 2, 3, 1)  # to HWC
    return images


def read_labels(label_file):
    with open(label_file, "rb") as f:
        labels = np.fromfile(f, dtype=np.uint8)
    return labels


def save_images(images, labels, split):
    for idx, (img, label) in enumerate(zip(images, labels)):
        class_name = CLASS_NAMES[label - 1]  # labels are 1-indexed
        class_dir = OUTPUT_DIR / split / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        img_pil = Image.fromarray(img)
        img_pil.save(class_dir / f"{class_name}_{split}_{idx:05d}.png")


def convert_to_images():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Converting training images...")
    train_images = read_images(EXTRACT_DIR / "train_X.bin")
    train_labels = read_labels(EXTRACT_DIR / "train_y.bin")
    save_images(train_images, train_labels, "train")

    print("Converting test images...")
    test_images = read_images(EXTRACT_DIR / "test_X.bin")
    test_labels = read_labels(EXTRACT_DIR / "test_y.bin")
    save_images(test_images, test_labels, "test")

    print("Conversion complete.")


archive = download_dataset()
extract_dataset(archive)
convert_to_images()