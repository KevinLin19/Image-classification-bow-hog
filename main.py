# main.py
from __future__ import annotations

from pathlib import Path

from src.experiments import run_multiclasses
from src.model import train_multiclass_model, save_model, load_model

DATA_ROOT = "data/stl10_raw"
TRAIN_ROOT = f"{DATA_ROOT}/train"
TEST_ROOT = f"{DATA_ROOT}/test"
MODELS_DIR = Path("outputs/models")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    ans = input(prompt + suffix).strip().lower()
    if ans == "":
        return default
    return ans in {"y", "yes", "o", "oui"}


def safe_int(prompt: str, default: int) -> int:
    ans = input(f"{prompt} (default={default}): ").strip()
    return default if ans == "" else int(ans)


def safe_metric(prompt: str, default: str = "l2") -> str:
    while True:
        ans = input(f"{prompt} (l1/l2, default={default}): ").strip().lower()
        if ans == "":
            return default
        if ans in {"l1", "l2"}:
            return ans
        print("❌ Invalid metric. Choose 'l1' or 'l2'.")


def default_model_path(class_names: list[str], K: int, metric: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_".join(class_names) if len(class_names) <= 3 else f"{len(class_names)}classes"
    return MODELS_DIR / f"{tag}_K{K}_{metric}.npz"


def list_saved_models() -> list[Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(MODELS_DIR.glob("*.npz"))


def choose_model_interactively() -> str | None:
    models = list_saved_models()
    if not models:
        print("\n❌ No saved models found in outputs/models/")
        if ask_yes_no("Enter a model path manually?", default=True):
            p = input("Enter model path: ").strip()
            return p if p else None
        return None

    print("\nAvailable saved models:")
    for i, p in enumerate(models, start=1):
        print(f"{i} → {p.name}")
    print("0 → Cancel")

    while True:
        c = input("Select a model: ").strip().lower()
        if c == "0":
            return None
        if c.isdigit():
            idx = int(c)
            if 1 <= idx <= len(models):
                return str(models[idx - 1])
        print("❌ Invalid choice.")


def run_experiment(class_names: list[str], K: int, runs: int, metric: str, npx: int, npy: int) -> None:
    print(f"\n▶ Running classes: {class_names}")
    print(f"   K={K}, runs={runs}, metric={metric}, grid={npx}x{npy}\n")

    mean_acc, std_acc = run_multiclasses(
        train_root=TRAIN_ROOT,
        test_root=TEST_ROOT,
        class_names=class_names,
        K=K,
        runs=runs,
        metric=metric,
        n_points_x=npx,
        n_points_y=npy,
        seed=0,
    )
    print(f"✅ Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")

    if ask_yes_no("Save a trained model for later predictions?", default=False):
        print("▶ Training one reproducible model to save (seed=0)...")
        model = train_multiclass_model(
            train_root=TRAIN_ROOT,
            class_names=class_names,
            K=K,
            metric=metric,
            n_points_x=npx,
            n_points_y=npy,
            seed=0,
        )
        path = default_model_path(class_names, K, metric)
        custom = input(f"Model save path (Enter to keep default: {path}): ").strip()
        save_path = custom if custom else str(path)
        save_model(model, save_path)
        print(f"✅ Model saved to: {save_path}\n")


def fast_bird_vs_airplane():
    run_experiment(["bird", "airplane"], K=50, runs=1, metric="l2", npx=8, npy=8)


def slow_bird_vs_airplane():
    run_experiment(["bird", "airplane"], K=200, runs=10, metric="l2", npx=8, npy=8)


def custom_classes():
    print("\n▶ Custom experiment (multi-class)")
    raw = input("Enter classes (comma-separated) or 'all' for 10 classes: ").strip().lower()
    if raw == "all":
        class_names = ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]
    else:
        class_names = [c.strip() for c in raw.split(",") if c.strip()]
        if len(class_names) < 2:
            print("❌ Please provide at least 2 classes.\n")
            return

    K = safe_int("Number of visual words K", 200)
    runs = safe_int("Number of runs", 10)
    metric = safe_metric("Distance metric", "l2")
    npx = safe_int("Grid points in x", 8)
    npy = safe_int("Grid points in y", 8)

    run_experiment(class_names, K=K, runs=runs, metric=metric, npx=npx, npy=npy)


def predict_with_model():
    print("\n▶ Predict with a saved model (.npz)")
    model_path = choose_model_interactively()
    if model_path is None:
        print("Cancelled.\n")
        return

    model = load_model(model_path)

    raw = input(
        "Enter image path (file), folder path, or multiple files separated by ';':\n> "
    ).strip()

    if not raw:
        print("❌ Empty input.\n")
        return

    parts = [p.strip().strip('"') for p in raw.split(";") if p.strip()]
    paths = []

    # If user provided a single folder
    if len(parts) == 1 and Path(parts[0]).exists() and Path(parts[0]).is_dir():
        folder = Path(parts[0])
        exts = (".png", ".jpg", ".jpeg")
        paths = sorted([str(x) for x in folder.iterdir() if x.suffix.lower() in exts])
        if not paths:
            print("❌ No images found in this folder.\n")
            return
        print(f"📂 Folder detected: {folder} ({len(paths)} images)")

    else:
        # Multiple files or single file
        for p in parts:
            pp = Path(p)
            if not pp.exists():
                print(f"❌ Path not found: {p}")
                return
            if pp.is_dir():
                print(f"❌ '{p}' is a folder. If you want a folder prediction, pass only that folder path.")
                return
            paths.append(str(pp))

    # Predict
    if len(paths) == 1:
        pred = model.predict_label(paths[0])
        print(f"\n✅ Model: {Path(model_path).name}")
        print(f"✅ Image: {Path(paths[0]).name}")
        print(f"🎯 Prediction: {pred}\n")
        return

    # Multiple images → show per-image results + optional vote
    print("\nPer-image predictions:")
    results = model.predict_many(paths)
    for p, lab in results:
        print(f"- {Path(p).name} → {lab}")

    if ask_yes_no("\nDo you want a final label using majority vote?", default=True):
        final, counts = model.predict_with_vote(paths)
        print("\nVote summary:")
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"- {k}: {v}")
        print(f"\n🎯 Final prediction (vote): {final}\n")


def main():
    while True:
        print("====================================")
        print(" STL-10 BoW + HOG + K-means + 1-NN ")
        print("====================================")
        print("1 → Bird vs Airplane (FAST)")
        print("2 → Bird vs Airplane (SLOW)")
        print("3 → Custom classes (2..10 or 'all')")
        print("4 → Predict using a saved model")
        print("0 → Exit")

        choice = input("\nSelect an option: ").strip()
        if choice == "1":
            fast_bird_vs_airplane()
        elif choice == "2":
            slow_bird_vs_airplane()
        elif choice == "3":
            custom_classes()
        elif choice == "4":
            predict_with_model()
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("❌ Invalid option.\n")


if __name__ == "__main__":
    main()