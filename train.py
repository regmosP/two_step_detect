"""Train the global or hard-defect specialist YOLO11 model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML")
    parser.add_argument("--weights", default="yolo11m.pt", help="Pretrained .pt or YOLO11 model yaml")
    parser.add_argument("--name", required=True, help="Run name")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", choices=("false", "ram", "disk"), default="false")
    parser.add_argument("--optimizer", choices=("AdamW", "SGD", "auto"), default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--freeze", type=int, default=0, help="Freeze first N layers; 0 trains all")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.data).is_file():
        raise SystemExit(f"Dataset YAML not found: {args.data}")
    model = YOLO(args.weights)
    train_args = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        cache=False if args.cache == "false" else args.cache,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=True,
        close_mosaic=args.close_mosaic,
        amp=args.amp,
        freeze=args.freeze if args.freeze > 0 else None,
        resume=args.resume,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
        save=True,
        save_period=10,
        val=True,
        # Mild geometric and color changes model normal test-bench variation.
        # Mosaic is turned off near the end so final epochs match inference images.
        hsv_h=0.015,
        hsv_s=0.35,
        hsv_v=0.25,
        degrees=5.0,
        translate=0.08,
        scale=0.35,
        shear=1.0,
        perspective=0.0002,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.7,
        mixup=0.05,
        copy_paste=0.0,
    )
    if args.no_augment:
        train_args.update(
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0,
            translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
            fliplr=0.0, flipud=0.0, mosaic=0.0, mixup=0.0,
        )
    model.train(**train_args)


if __name__ == "__main__":
    main()
