"""Create a YOLO dataset transformed with the same preprocessing used at inference."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

from cookie_defect.preprocessing import preprocess_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="YOLO images root, containing train/ and val/")
    parser.add_argument("--labels", required=True, help="Mirrored YOLO labels root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=("clahe", "binary", "enhanced"))
    parser.add_argument("--class-names", required=True, help="Names in class-id order")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    image_root, label_root, output = Path(args.images), Path(args.labels), Path(args.output)
    if not image_root.is_dir() or not label_root.is_dir():
        raise SystemExit("--images and --labels must be existing directories")
    if output.exists():
        if not args.force:
            raise SystemExit(f"Output exists; pass --force to replace it: {output}")
        shutil.rmtree(output)
    count = 0
    for source in sorted(p for p in image_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        relative = source.relative_to(image_root)
        image = cv2.imread(str(source))
        if image is None:
            continue
        destination = output / "images" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), preprocess_image(image, mode=args.mode)):
            raise SystemExit(f"Could not write processed image: {destination}")
        count += 1
    if count == 0:
        raise SystemExit("No readable images found")
    shutil.copytree(label_root, output / "labels")
    names = [name.strip() for name in args.class_names.split(",")]
    yaml_lines = [f"path: {output.resolve().as_posix()}", "train: images/train", "val: images/val", "names:"]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"Processed {count} images using {args.mode}; dataset config: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
