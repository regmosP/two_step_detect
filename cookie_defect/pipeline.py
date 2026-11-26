"""Two-stage YOLO11 detection with scale/flip TTA and box fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
from ultralytics import YOLO

from .preprocessing import preprocess_image


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _fuse(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Greedy class-aware weighted box fusion over TTA predictions."""
    fused: list[Detection] = []
    for class_id in sorted({d.class_id for d in detections}):
        remaining = sorted((d for d in detections if d.class_id == class_id),
                           key=lambda d: d.confidence, reverse=True)
        while remaining:
            seed = remaining.pop(0)
            seed_box = np.array([seed.x1, seed.y1, seed.x2, seed.y2], dtype=float)
            cluster = [seed]
            keep = []
            for candidate in remaining:
                box = np.array([candidate.x1, candidate.y1, candidate.x2, candidate.y2], dtype=float)
                (cluster if _iou(seed_box, box) >= iou_threshold else keep).append(candidate)
            remaining = keep
            weights = np.array([max(d.confidence, 1e-6) for d in cluster])
            boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in cluster])
            box = np.average(boxes, axis=0, weights=weights)
            best = max(cluster, key=lambda d: d.confidence)
            # TTA views are correlated; probabilistic OR inflates confidence as
            # views are added. Keep the fused score on the model's score scale.
            confidence = float(np.average([d.confidence for d in cluster], weights=weights))
            fused.append(Detection(*map(float, box), confidence, class_id, best.class_name))
    return sorted(fused, key=lambda d: d.confidence, reverse=True)


class TwoStageDetector:
    """Global YOLO followed by a specialist YOLO on configured hard classes."""

    DEFAULT_SPECIALIST_CLASSES = ("缺角", "破碎")

    def __init__(
        self,
        global_weights: str | Path,
        specialist_weights: str | Path,
        *,
        specialist_classes: Iterable[str] = DEFAULT_SPECIALIST_CLASSES,
        specialist_class_map: Mapping[str, str] | None = None,
        confidence: float = 0.25,
        global_candidate_confidence: float = 0.10,
        specialist_confidence: float = 0.10,
        class_confidence: Mapping[str, float] | None = None,
        iou: float = 0.5,
        image_size: int = 640,
        tta_scales: Iterable[float] = (1.0, 0.85, 1.15),
        tta_flips: Iterable[str] = ("none", "horizontal"),
        device: str | None = None,
        preprocess_mode: str = "raw",
    ) -> None:
        self.global_model = YOLO(str(global_weights))
        self.specialist_model = YOLO(str(specialist_weights))
        self.specialist_classes = set(specialist_classes)
        self.class_map = dict(specialist_class_map or {})
        self.confidence = confidence
        self.global_candidate_confidence = global_candidate_confidence
        self.specialist_confidence = specialist_confidence
        self.class_confidence = dict(class_confidence or {})
        self.iou, self.image_size = iou, image_size
        self.tta_scales = tuple(float(s) for s in tta_scales)
        self.tta_flips = tuple(tta_flips)
        self.device, self.preprocess_mode = device, preprocess_mode
        if preprocess_mode not in {"raw", "clahe", "binary", "enhanced"}:
            raise ValueError("preprocess_mode must be raw, clahe, binary, or enhanced")
        if not self.tta_scales or any(s <= 0 for s in self.tta_scales):
            raise ValueError("tta_scales must contain positive values")
        if any(f not in {"none", "horizontal", "vertical"} for f in self.tta_flips):
            raise ValueError("tta_flips supports none, horizontal, and vertical")

    def _predict_tta(self, model: YOLO, image: np.ndarray, confidence: float) -> list[Detection]:
        height, width = image.shape[:2]
        all_detections: list[Detection] = []
        for scale in self.tta_scales:
            resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
            for flip in self.tta_flips:
                transformed = resized
                if flip == "horizontal":
                    transformed = cv2.flip(resized, 1)
                elif flip == "vertical":
                    transformed = cv2.flip(resized, 0)
                result = model.predict(transformed, conf=confidence, iou=self.iou,
                                       imgsz=self.image_size, device=self.device, verbose=False)[0]
                if result.boxes is None:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                names = result.names
                for box, score, cls_id in zip(boxes, scores, class_ids):
                    x1, y1, x2, y2 = box.astype(float) / scale
                    if flip == "horizontal":
                        x1, x2 = width - x2, width - x1
                    elif flip == "vertical":
                        y1, y2 = height - y2, height - y1
                    x1, x2 = np.clip([x1, x2], 0, width)
                    y1, y2 = np.clip([y1, y2], 0, height)
                    all_detections.append(Detection(float(x1), float(y1), float(x2), float(y2),
                                                    float(score), int(cls_id), str(names[int(cls_id)])))
        return _fuse(all_detections, self.iou)

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Predict on a BGR image and return boxes in its original coordinates."""
        model_image = preprocess_image(image, mode=self.preprocess_mode)
        global_dets = self._predict_tta(self.global_model, model_image, self.global_candidate_confidence)
        hard = [d for d in global_dets if d.class_name in self.specialist_classes]
        other = [d for d in global_dets if d.class_name not in self.specialist_classes]
        height, width = model_image.shape[:2]
        refined: list[Detection] = []
        fallback: list[Detection] = []
        for candidate in hard:
            # Expand candidate context so the specialist can inspect the defect
            # and nearby cookie edge, then map crop predictions to full image.
            pad_x = max(4.0, (candidate.x2 - candidate.x1) * 0.25)
            pad_y = max(4.0, (candidate.y2 - candidate.y1) * 0.25)
            x1, y1 = max(0, int(candidate.x1 - pad_x)), max(0, int(candidate.y1 - pad_y))
            x2, y2 = min(width, int(candidate.x2 + pad_x)), min(height, int(candidate.y2 + pad_y))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = model_image[y1:y2, x1:x2]
            crop_dets = self._predict_tta(self.specialist_model, crop, self.specialist_confidence)
            candidate_refined = False
            for det in crop_dets:
                mapped_name = self.class_map.get(det.class_name, det.class_name)
                # A specialist with a different vocabulary must be explicitly mapped.
                global_id = next((i for i, name in self.global_model.names.items()
                                  if str(name) == mapped_name), None)
                if global_id is None:
                    continue
                refined.append(Detection(det.x1 + x1, det.y1 + y1, det.x2 + x1, det.y2 + y1,
                                         det.confidence, int(global_id), mapped_name))
                candidate_refined = True
            # If this proposal yields no valid specialist result, retain it.
            if not candidate_refined:
                fallback.append(candidate)
        results = other + refined + fallback
        results = [d for d in results
                   if d.confidence >= self.class_confidence.get(d.class_name, self.confidence)]
        return sorted(results, key=lambda d: d.confidence, reverse=True)
