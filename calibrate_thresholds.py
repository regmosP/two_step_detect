"""Choose per-class confidence thresholds from held-out YOLO validation data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="JSON directory produced by infer.py")
    parser.add_argument("--images", required=True, help="Held-out validation images")
    parser.add_argument("--labels", required=True, help="YOLO validation labels")
    parser.add_argument("--class-names", required=True, help="Names in class-id order")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--max-conf", type=float, default=0.95)
    parser.add_argument("--steps", type=int, default=19)
    parser.add_argument("--output", default="thresholds.csv")
    args = parser.parse_args()

    images, labels, pred_dir = Path(args.images), Path(args.labels), Path(args.predictions)
    names = [name.strip() for name in args.class_names.split(",")]
    gt_by_image = {}
    pred_by_image = {}
    for image_path in sorted(p for p in images.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        rel = image_path.relative_to(images)
        label_path = labels / rel.with_suffix(".txt")
        if not label_path.is_file():
            label_path = labels / f"{image_path.stem}.txt"
        prediction_path = pred_dir / f"{image_path.stem}.json"
        if not label_path.is_file() or not prediction_path.is_file():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        ground_truth = defaultdict(list)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            cid = int(float(fields[0]))
            if not 0 <= cid < len(names):
                continue
            cx, cy, bw, bh = map(float, fields[1:5])
            ground_truth[names[cid]].append(((cx-bw/2)*width, (cy-bh/2)*height,
                                               (cx+bw/2)*width, (cy+bh/2)*height))
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        gt_by_image[image_path] = dict(ground_truth)
        pred_by_image[image_path] = predictions
    if not gt_by_image:
        raise SystemExit("No matching image, label, and prediction JSON triplets found")

    thresholds = [args.min_conf + (args.max_conf - args.min_conf) * i / max(1, args.steps - 1)
                  for i in range(args.steps)]
    rows = []
    for class_name in names:
        total_gt = sum(len(gt.get(class_name, [])) for gt in gt_by_image.values())
        if total_gt == 0:
            continue
        for threshold in thresholds:
            tp = fp = 0
            for image_path, gt in gt_by_image.items():
                truths = gt.get(class_name, [])
                matched = set()
                candidates = sorted(
                    (p for p in pred_by_image[image_path]
                     if p.get("class_name") == class_name and p.get("confidence", 0) >= threshold),
                    key=lambda p: p.get("confidence", 0), reverse=True,
                )
                for prediction in candidates:
                    box = (prediction["x1"], prediction["y1"], prediction["x2"], prediction["y2"])
                    best_iou, best_idx = 0.0, None
                    for idx, truth in enumerate(truths):
                        score = iou(box, truth)
                        if idx not in matched and score > best_iou:
                            best_iou, best_idx = score, idx
                    if best_idx is not None and best_iou >= args.iou:
                        tp += 1
                        matched.add(best_idx)
                    else:
                        fp += 1
            fn = total_gt - tp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / total_gt if total_gt else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append({"class_name": class_name, "threshold": threshold, "precision": precision,
                         "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn})

    with Path(args.output).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["class_name", "threshold", "precision", "recall", "f1", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(rows)
    for name in names:
        class_rows = [row for row in rows if row["class_name"] == name]
        if class_rows:
            best = max(class_rows, key=lambda row: row["f1"])
            print(f"{name}: threshold={best['threshold']:.3f}, precision={best['precision']:.3f}, "
                  f"recall={best['recall']:.3f}, F1={best['f1']:.3f}")
    print(f"Wrote threshold sweep: {args.output}")


if __name__ == "__main__":
    main()
