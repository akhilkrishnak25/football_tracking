"""
train.py
Script to fine-tune a YOLO11 model on a custom sports ball dataset.
Saves the resulting best weights to models/best.pt.
"""

import os
import sys
import shutil
import argparse
import torch
from ultralytics import YOLO


def check_dataset_readiness(data_yaml_path: str) -> bool:
    """Validate that the dataset YAML and image directories exist."""
    if not os.path.exists(data_yaml_path):
        print(f"[ERROR] Dataset configuration file not found at: {data_yaml_path}")
        return False
    return True


def train_custom_ball_model(
    model_name: str = "yolo11n.pt",
    data_yaml: str = "dataset/data.yaml",
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "auto",
    output_model_path: str = "models/best.pt"
):
    """
    Train YOLO11 model on the sports ball dataset and save best weights.
    """
    print("=" * 65)
    print("⚽ Sports Ball YOLO11 Model Training")
    print(f"Base Model:       {model_name}")
    print(f"Dataset Config:   {data_yaml}")
    print(f"Epochs:           {epochs}")
    print(f"Batch Size:       {batch_size}")
    print(f"Image Size:       {imgsz}")
    print("=" * 65)

    if not check_dataset_readiness(data_yaml):
        print("\nPlease make sure images and annotations are placed in 'dataset/' before training.")
        sys.exit(1)

    # Determine device
    if device == "auto":
        resolved_device = "0" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    print(f"[INFO] Using hardware device: {resolved_device}")

    # Load base model
    print(f"[INFO] Loading base model: {model_name}...")
    model = YOLO(model_name)

    # Train
    print("[INFO] Initiating training run...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=resolved_device,
        project="runs/train",
        name="ball_model",
        exist_ok=True
    )

    # Check for best.pt in runs/train/ball_model/weights/best.pt
    trained_best = os.path.join("runs", "train", "ball_model", "weights", "best.pt")
    if os.path.exists(trained_best):
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        shutil.copy2(trained_best, output_model_path)
        print("=" * 65)
        print(f"[SUCCESS] Training complete!")
        print(f"[SUCCESS] Best model weights successfully copied to: {output_model_path}")
        print("=" * 65)
    else:
        print(f"[WARNING] Finished training, but '{trained_best}' was not found.")
        print("Check Ultralytics training run directory for outputs.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11 on Custom Sports Ball Dataset")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model (yolo11n.pt, yolo11s.pt, yolo11m.pt)")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for training")
    parser.add_argument("--device", type=str, default="auto", help="Hardware device ('auto', 'cpu', '0')")
    parser.add_argument("--output", type=str, default="models/best.pt", help="Path to save best model weights")

    args = parser.parse_args()

    train_custom_ball_model(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        output_model_path=args.output
    )


if __name__ == "__main__":
    main()
