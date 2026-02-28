# STL-10 Image Classification from Scratch  
**HOG + Bag-of-Words + K-means + 1-Nearest Neighbor**

This project implements a complete classical image classification pipeline from scratch using NumPy and SciPy.

It reproduces a traditional computer vision workflow:

Dense Grid → HOG descriptors → K-means (visual vocabulary)  
→ Bag-of-Words histograms → 1-Nearest Neighbor classification

No scikit-learn. No PyTorch. No TensorFlow.  
Everything is implemented manually for educational purposes.

---

# Features

- Multi-class classification (2 to 10 classes)
- Custom class selection
- Train & save models
- Load saved models
- Predict on:
  - Single image
  - Multiple images
  - Entire folder
  - Majority vote prediction
- Multiple runs for robustness
- Progress bars during training

---

# Dataset: STL-10

This project expects STL-10 to be stored as image files in class folders.

---

## Step 1 — Download STL-10

You can download it automatically:

```bash
python scripts/download_stl10.py
```

---

## Step 2 — Required Folder Structure

Create this structure at the root of the repository:

```text
data/
└── stl10_raw/
    ├── train/
    │   ├── airplane/
    │   ├── bird/
    │   ├── car/
    │   ├── cat/
    │   ├── deer/
    │   ├── dog/
    │   ├── horse/
    │   ├── monkey/
    │   ├── ship/
    │   └── truck/
    └── test/
        ├── airplane/
        ├── bird/
        ├── car/
        ├── cat/
        ├── deer/
        ├── dog/
        ├── horse/
        ├── monkey/
        ├── ship/
        └── truck/
```

Each folder must contain `.png`, `.jpg`, or `.jpeg` images.

---

# Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the Project

Start the interactive menu:

```bash
python main.py
```

You will see:

```text
1 → Bird vs Airplane (FAST)
2 → Bird vs Airplane (SLOW)
3 → Custom classes (2..10 or 'all')
4 → Predict using a saved model
0 → Exit
```

---

# Experiment Modes

## Fast Mode
- 2 classes
- K=50
- 1 run
- Quick test

## Slow Mode
- 2 classes
- K=200
- 10 runs
- More stable accuracy

## Custom Mode

You can:
- Select 2 to 10 classes
- Or type `all` for all 10 STL-10 classes
- Choose:
  - Number of visual words (K)
  - Number of runs
  - Distance metric (L1 or L2)
  - Grid resolution

---

# Saving Models

After training, you can choose to save the model.

A saved model contains:
- K-means centroids (visual vocabulary)
- Training BoW histograms
- Training labels
- Class names
- Grid configuration

Saved models are stored in:

```text
outputs/models/
```

Example filename:

```text
bird_airplane_K200_l2.npz
```

---

# Predicting New Images

You can predict:

## Single image

```text
C:\path\to\image.png
```

## Entire folder

```text
C:\path\to\folder\
```

All images inside will be predicted.

## Multiple images

```text
img1.png; img2.png; img3.png
```

You can also request a majority vote prediction across multiple images.

---

# Pipeline Details

## 1) Dense Grid Sampling
Uniform grid across the image.

## 2) HOG Descriptor
- 16×16 patches
- 4×4 cells
- 8 orientation bins
- 128-dimensional descriptor

## 3) K-means
- Learn visual vocabulary of size K
- Manual implementation
- Multiple runs supported

## 4) Bag-of-Words
Each image becomes a K-dimensional histogram.

## 5) 1-Nearest Neighbor
Brute-force distance comparison in BoW space.

---

# Complexity

Let:
- T = number of training images
- K = number of visual words

Space complexity:

```
O(T × K)
```

Time complexity (per test image):

```
O(T × K)
```

---

# Example Accuracy (Typical)

Binary classification (bird vs airplane):
- K=50 → ~60–70%
- K=200 → ~70–80%

10-class classification:
- Typically 40–60% depending on K

---

# Requirements

- numpy
- scipy
- pillow
- matplotlib
- tqdm

---

# Educational Goal

This project demonstrates:
- Classical computer vision pipelines
- Feature extraction with HOG
- Unsupervised learning (K-means)
- Nearest neighbor classification
- Model serialization
- Experimental evaluation

---

## Author

**Kevin Lin**  
AI & Software Engineering

Built as part of a personal exploration of classical computer vision and machine learning from scratch.
