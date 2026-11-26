"""Preprocessing for cookie images captured under variable bench lighting."""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(
    image: np.ndarray,
    *,
    mode: str = "enhanced",
    denoise_strength: int = 7,
    adaptive_block_size: int = 31,
    adaptive_c: int = 5,
    morph_kernel_size: int = 3,
) -> np.ndarray:
    """Apply one of the preprocessing variants used for validation ablations.

    The adaptive binary mask suppresses background/illumination variation. It is
    combined with a local-contrast grayscale image and returned as BGR so it can
    be passed to a standard three-channel YOLO model.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image
    else:
        raise ValueError("Expected a grayscale or BGR image")
    if mode == "raw":
        return bgr.copy()
    if mode not in {"enhanced", "clahe", "binary"}:
        raise ValueError("mode must be raw, clahe, binary, or enhanced")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Bilateral filtering removes sensor noise without washing out crack edges.
    filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=denoise_strength * 4,
                                   sigmaSpace=denoise_strength)
    local_contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(filtered)

    block = max(3, int(adaptive_block_size) | 1)
    binary = cv2.adaptiveThreshold(
        local_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, adaptive_c,
    )
    kernel_size = max(1, int(morph_kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if mode == "clahe":
        return cv2.cvtColor(local_contrast, cv2.COLOR_GRAY2BGR)
    if mode == "binary":
        return cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
    # Preserve grayscale texture while using the binary result to normalize the
    # foreground/background response; this avoids feeding a harsh binary image.
    enhanced = cv2.addWeighted(local_contrast, 0.72, clean, 0.28, 0)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
