"""Image preprocessing aligned with the SWRD training experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessResult:
    image: np.ndarray
    elapsed_ms: float


def apply_preprocess(image_rgb: np.ndarray, mode: str) -> PreprocessResult:
    start = time.perf_counter()
    mode = (mode or 'none').lower()
    if mode == 'clahe':
        image = _clahe(image_rgb)
    elif mode == 'gaussian':
        image = cv2.GaussianBlur(image_rgb, (5, 5), 0)
    elif mode == 'gamma':
        image = _adjust_gamma(image_rgb, 1.5)
    else:
        image = image_rgb.copy()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PreprocessResult(image=image, elapsed_ms=elapsed_ms)


def _clahe(image_rgb: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(image_rgb)
    processed = [clahe.apply(channel) for channel in channels]
    return cv2.merge(processed)


def _adjust_gamma(image_rgb: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0)**inv_gamma * 255
                      for i in np.arange(256)]).astype('uint8')
    return cv2.LUT(image_rgb, table)

