"""
Train a YOLOv8 Minutiae Detector on Your Verified Dataset
==============================================================
Run this AFTER you have manually verified/corrected the draft labels
produced by ``bootstrap_labels.py`` and split them into train/val sets.

Expected folder layout (standard Ultralytics YOLO format):

    data/minutiae_dataset/
        images/train/*.png
        images/val/*.png
        labels/train/*.txt
        labels/val/*.txt

Usage:
    pip install ultralytics
    python tools/train_yolo.py --data_dir data/minutiae_dataset --epochs 100

The best checkpoint is automatically copied to models/minutiae_yolo.pt,
which app.py looks for at startup — no other code changes needed.
"""

import argparse
import os
import shutil
import sys

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

DATASET_YAML_TEMPLATE = """\
path: {data_dir}
train: images/train
val: images/val

names:
  0: ridge_ending
  1: bifurcation
"""


def write_dataset_yaml(data_dir: str) -> str:
    yaml_path = os.path.join(data_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(DATASET_YAML_TEMPLATE.format(data_dir=os.path.abspath(data_dir)))
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the verified minutiae dataset.")
    parser.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "data", "minutiae_dataset"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--base_model", default="yolov8n.pt",
                         help="Small base checkpoint — fine for a limited, single-domain dataset like this.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Run: pip install ultralytics --break-system-packages")
        return

    if not os.path.isdir(os.path.join(args.data_dir, "images", "train")):
        print(f"Expected dataset at {args.data_dir}/images/train (and val) — see this "
              "script's docstring for the required folder layout.")
        return

    yaml_path = write_dataset_yaml(args.data_dir)
    print(f"Dataset config written to {yaml_path}")

    model = YOLO(args.base_model)
    results = model.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz,
                           project=os.path.join(PROJECT_ROOT, "runs"), name="minutiae_yolo")

    best_weights = os.path.join(results.save_dir, "weights", "best.pt")
    target_path = os.path.join(PROJECT_ROOT, "models", "minutiae_yolo.pt")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy(best_weights, target_path)

    print(f"\nTraining complete. Best weights copied to: {target_path}")
    print("app.py will automatically use this model the next time it runs "
          "(toggle is in the sidebar under 'Feature Detection').")


if __name__ == "__main__":
    main()
