from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from cookie_defect import TwoStageDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage cookie defect detection")
    parser.add_argument("--image", required=True, help="Image file or directory")
    parser.add_argument("--global-weights", required=True)
    parser.add_argument("--specialist-weights", required=True)
    parser.add_argument("--output", default="runs/predictions")
    parser.add_argument("--specialist-classes", default="缺角,破碎")
    parser.add_argument("--specialist-class-map", default="缺角:缺角,破碎:破碎")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--global-candidate-conf", type=float, default=0.10,
                        help="Low first-stage threshold to preserve recall before specialist review")
    parser.add_argument("--specialist-conf", type=float, default=0.10)
    parser.add_argument("--class-conf", default="",
                        help="Per-class output thresholds as class:score pairs separated by commas")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tta-scales", default="1.0,0.85,1.15")
    parser.add_argument("--tta-flips", default="none,horizontal")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--preprocess", choices=("raw", "clahe", "binary", "enhanced"), default="raw",
                        help="Use the same preprocessing variant as the training data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if image_path.is_dir():
        supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        paths = sorted(p for p in image_path.iterdir() if p.suffix.lower() in supported)
    else:
        paths = [image_path]
    if not paths:
        raise SystemExit(f"No supported images found: {image_path}")

    class_map = {}
    for pair in filter(None, args.specialist_class_map.split(",")):
        specialist_name, global_name = pair.split(":", 1)
        class_map[specialist_name.strip()] = global_name.strip()
    class_conf = {}
    for pair in filter(None, args.class_conf.split(",")):
        class_name, threshold = pair.split(":", 1)
        class_conf[class_name.strip()] = float(threshold)
    detector = TwoStageDetector(
        args.global_weights, args.specialist_weights,
        specialist_classes=[x.strip() for x in args.specialist_classes.split(",") if x.strip()],
        specialist_class_map=class_map, confidence=args.conf, iou=args.iou,
        global_candidate_confidence=args.global_candidate_conf,
        specialist_confidence=args.specialist_conf,
        class_confidence=class_conf,
        image_size=args.imgsz, tta_scales=[float(x) for x in args.tta_scales.split(",")],
        tta_flips=[x.strip() for x in args.tta_flips.split(",")],
        device=args.device, preprocess_mode=args.preprocess,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skip unreadable image: {path}")
            continue
        detections = detector.predict(image)
        for det in detections:
            p1, p2 = (round(det.x1), round(det.y1)), (round(det.x2), round(det.y2))
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.rectangle(image, p1, p2, (40, 220, 40), 2)
            cv2.putText(image, label, (p1[0], max(18, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / path.name), image)
        with (output_dir / f"{path.stem}.json").open("w", encoding="utf-8") as f:
            json.dump([d.to_dict() for d in detections], f, ensure_ascii=False, indent=2)
        print(f"{path}: {len(detections)} detections")


if __name__ == "__main__":
    main()
