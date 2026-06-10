"""Shared constants for the SWRD segmentation GUI."""

from __future__ import annotations

from dataclasses import dataclass


SWRD_CLASSES = (
    'background',
    'air-hole',
    'bite-edge',
    'broken-arc',
    'crack',
    'hollow-bead',
    'overlap',
    'slag-inclusion',
    'unfused',
)

SWRD_PALETTE = (
    (0, 0, 0),
    (220, 20, 60),
    (255, 127, 14),
    (44, 160, 44),
    (31, 119, 180),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (188, 189, 34),
)

PREPROCESS_OPTIONS = ('none', 'clahe', 'gaussian', 'gamma')
DEFAULT_INPUT_SIZE = (512, 512)


@dataclass(frozen=True)
class TimingInfo:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def fps(self) -> float:
        total = self.inference_ms + self.postprocess_ms
        if total <= 0:
            return 0.0
        return 1000.0 / total
