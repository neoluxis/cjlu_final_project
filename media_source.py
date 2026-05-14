"""Single image, image-folder, and video-frame media sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


MediaKind = Literal['none', 'image', 'folder', 'video']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}


@dataclass(frozen=True)
class MediaFrame:
    image_rgb: np.ndarray
    label: str
    index: int
    total: int
    kind: MediaKind


class MediaSource:
    def __init__(self) -> None:
        self.kind: MediaKind = 'none'
        self.paths: list[Path] = []
        self.video_path: Path | None = None
        self.capture: cv2.VideoCapture | None = None
        self.index = 0
        self.total = 0

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None

    def open_image(self, path: str | Path) -> MediaFrame:
        self.close()
        image_path = Path(path)
        self.kind = 'image'
        self.paths = [image_path]
        self.video_path = None
        self.index = 0
        self.total = 1
        return self.current()

    def open_folder(self, folder: str | Path) -> MediaFrame:
        self.close()
        root = Path(folder)
        paths = [
            p for p in sorted(root.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not paths:
            raise ValueError(f'No supported images found in {root}')
        self.kind = 'folder'
        self.paths = paths
        self.video_path = None
        self.index = 0
        self.total = len(paths)
        return self.current()

    def open_video(self, path: str | Path) -> MediaFrame:
        self.close()
        video_path = Path(path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f'Could not open video: {video_path}')
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.kind = 'video'
        self.paths = []
        self.video_path = video_path
        self.capture = capture
        self.index = 0
        self.total = max(total, 1)
        return self.current()

    def current(self) -> MediaFrame:
        if self.kind in {'image', 'folder'}:
            path = self.paths[self.index]
            image = _read_image_rgb(path)
            return MediaFrame(
                image_rgb=image,
                label=path.name,
                index=self.index,
                total=self.total,
                kind=self.kind,
            )
        if self.kind == 'video':
            if self.capture is None or self.video_path is None:
                raise ValueError('Video is not open')
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.index)
            ok, frame_bgr = self.capture.read()
            if not ok:
                raise ValueError(f'Could not read frame {self.index}')
            image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return MediaFrame(
                image_rgb=image,
                label=f'{self.video_path.name} frame {self.index + 1}',
                index=self.index,
                total=self.total,
                kind=self.kind,
            )
        raise ValueError('No media source is open')

    def next(self) -> MediaFrame:
        if self.total <= 0:
            raise ValueError('No media source is open')
        self.index = min(self.index + 1, self.total - 1)
        return self.current()

    def previous(self) -> MediaFrame:
        if self.total <= 0:
            raise ValueError('No media source is open')
        self.index = max(self.index - 1, 0)
        return self.current()


def _read_image_rgb(path: Path) -> np.ndarray:
    frame_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError(f'Could not read image: {path}')
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

