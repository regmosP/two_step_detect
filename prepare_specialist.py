"""Build a crop-based YOLO dataset for hard classes such as 缺角 and 破碎."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--labels", required=True, help="YOLO label directory")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--class-names", required=True, help="Original dataset names, comma-separated in class-id order")
    parser.add_argument("--target-classes", default="缺角,破碎", help="Classes to keep in specialist labels")
    parser.add_argument("--padding", type=float, default=0.5, help="Crop context as fraction of target box size")
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Used only when images are not already in train/val subfolders")
    parser.add_argument("--force", action="store_true", help="Replace existing output directory")
    return parser.parse_args()


def image_label_pairs(root: Path, labels: Path):
    for image_path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        rel = image_path.relative_to(root)
        # Accept both mirrored split folders (images/train/a.jpg and
        # labels/train/a.txt) and a flat annotation directory.
        label_path = labels / rel.with_suffix(".txt")
        if not label_path.is_file():
            label_path = labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        first = rel.parts[0].lower() if len(rel.parts) > 1 else ""
        split = "val" if first in {"val", "valid", "validation"} else "train"
        yield image_path, label_path, split


def main() -> None:
    args = parse_args()
    image_root, label_root, output = Path(args.images), Path(args.labels), Path(args.output)
    if not image_root.is_dir() or not label_root.is_dir():
        raise SystemExit("Both --images and --labels must be existing directories")
    if output.exists():
        if not args.force:
            raise SystemExit(f"Output exists; pass --force to replace it: {output}")
        shutil.rmtree(output)

    names = [x.strip() for x in args.class_names.split(",")]
    targets = [x.strip() for x in args.target_classes.split(",") if x.strip()]
    missing = set(targets) - set(names)
    if missing:
        raise SystemExit(f"Target class names not in --class-names: {', '.join(sorted(missing))}")
    old_to_new = {names.index(name): i for i, name in enumerate(targets)}
    pairs = list(image_label_pairs(image_root, label_root))
    if not pairs:
        raise SystemExit("No image/label pairs found")

    # When source data has no explicit split, split by source image before
    # cropping so near-identical crops cannot leak across train and validation.
    has_explicit_val = any(split == "val" for _, _, split in pairs)
    if not has_explicit_val:
        rng = random.Random(args.seed)
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * args.val_fraction)) if len(shuffled) > 1 else 0
        val_sources = {p[0] for p in shuffled[:val_count]}
        pairs = [(img, lab, "val" if img in val_sources else "train") for img, lab, _ in pairs]

    counts = {"train": 0, "val": 0}
    for image_path, label_path, split in pairs:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        rows = []
        non_target_rows = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            class_id = int(float(fields[0]))
            cx, cy, bw, bh = map(float, fields[1:5])
            if class_id in old_to_new:
                rows.append((old_to_new[class_id], cx * width, cy * height, bw * width, bh * height))
            else:
                non_target_rows.append((cx * width, cy * height, bw * width, bh * height))
        for index, (new_id, cx, cy, bw, bh) in enumerate(rows):
            pad_x, pad_y = max(args.min_size / 2, bw * args.padding), max(args.min_size / 2, bh * args.padding)
            x1, y1 = max(0, int(cx - bw / 2 - pad_x)), max(0, int(cy - bh / 2 - pad_y))
            x2, y2 = min(width, int(cx + bw / 2 + pad_x)), min(height, int(cy + bh / 2 + pad_y))
            crop_w, crop_h = x2 - x1, y2 - y1
            if crop_w <= 0 or crop_h <= 0:
                continue
            out_img_dir = output / "images" / split
            out_lbl_dir = output / "labels" / split
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)
            relative_stem = image_path.relative_to(image_root).with_suffix("").as_posix().replace("/", "_").replace("\\", "_")
            stem = f"{relative_stem}_{index:03d}"
            cv2.imwrite(str(out_img_dir / f"{stem}.jpg"), image[y1:y2, x1:x2])
            # Keep every target-class object visible in the crop, not only its
            # center object, and convert coordinates to crop-relative YOLO form.
            crop_labels = []
            for other_id, ocx, ocy, obw, obh in rows:
                ox1, oy1 = max(ocx - obw / 2, x1), max(ocy - obh / 2, y1)
                ox2, oy2 = min(ocx + obw / 2, x2), min(ocy + obh / 2, y2)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                ncx, ncy = ((ox1 + ox2) / 2 - x1) / crop_w, ((oy1 + oy2) / 2 - y1) / crop_h
                nbw, nbh = (ox2 - ox1) / crop_w, (oy2 - oy1) / crop_h
                crop_labels.append(f"{other_id} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")
            (out_lbl_dir / f"{stem}.txt").write_text("\n".join(crop_labels) + "\n", encoding="utf-8")
            counts[split] += 1

        # Background/other-defect examples teach the specialist when not to
        # fire. Crop around a non-target object when possible so these are
        # harder negatives than an empty background-only image.
        if not rows:
            out_img_dir = output / "images" / split
            out_lbl_dir = output / "labels" / split
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)
            relative_stem = image_path.relative_to(image_root).with_suffix("").as_posix().replace("/", "_").replace("\\", "_")
            if non_target_rows:
                _, (ncx, ncy, nbw, nbh) = max(
                    enumerate(non_target_rows), key=lambda item: item[1][2] * item[1][3]
                )
                pad_x, pad_y = max(args.min_size / 2, nbw * args.padding), max(args.min_size / 2, nbh * args.padding)
                nx1, ny1 = max(0, int(ncx - nbw / 2 - pad_x)), max(0, int(ncy - nbh / 2 - pad_y))
                nx2, ny2 = min(width, int(ncx + nbw / 2 + pad_x)), min(height, int(ncy + nbh / 2 + pad_y))
                negative_crop = image[ny1:ny2, nx1:nx2]
            else:
                negative_crop = image
            cv2.imwrite(str(out_img_dir / f"{relative_stem}_negative.jpg"), negative_crop)
            (out_lbl_dir / f"{relative_stem}_negative.txt").write_text("", encoding="utf-8")
            counts[split] += 1

    if counts["train"] == 0:
        raise SystemExit("No specialist training crops were generated; check class IDs and labels")
    if counts["val"] == 0:
        raise SystemExit("No validation crops were generated; provide source images in a val/ folder or at least two images")
    yaml = [f"path: {output.resolve().as_posix()}",
            "train: images/train", "val: images/val", "names:"]
    yaml.extend(f"  {i}: {name}" for i, name in enumerate(targets))
    (output / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"Wrote {counts['train']} train and {counts['val']} val crops to {output}")
    print(f"Dataset config: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
