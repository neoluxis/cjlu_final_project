"""Persistent GUI settings."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QByteArray, QSettings


@dataclass
class AppSettings:
    model_dir: str = ''
    model_path: str = ''
    network_name: str = 'Unknown Network'
    preprocess: str = 'none'
    threshold: float = 0.5
    image_dir: str = ''
    image_path: str = ''
    video_path: str = ''
    media_kind: str = 'none'
    media_index: int = 0
    preview_mode: str = 'original'
    fill_overlay: bool = True
    geometry: QByteArray | None = None
    window_state: QByteArray | None = None


class SettingsStore:
    def __init__(self) -> None:
        self.settings = QSettings('neolux', 'segformer_app')

    def load(self) -> AppSettings:
        settings = self.settings
        return AppSettings(
            model_dir=settings.value('model_dir', '', str),
            model_path=settings.value('model_path', '', str),
            network_name=settings.value('network_name', 'Unknown Network', str),
            preprocess=settings.value('preprocess', 'none', str),
            threshold=float(settings.value('threshold', 0.5)),
            image_dir=settings.value('image_dir', '', str),
            image_path=settings.value('image_path', '', str),
            video_path=settings.value('video_path', '', str),
            media_kind=settings.value('media_kind', 'none', str),
            media_index=int(settings.value('media_index', 0)),
            preview_mode=settings.value('preview_mode', 'original', str),
            fill_overlay=_as_bool(settings.value('fill_overlay', True)),
            geometry=settings.value('geometry', None),
            window_state=settings.value('window_state', None),
        )

    def save(self, state: AppSettings) -> None:
        settings = self.settings
        settings.setValue('model_dir', state.model_dir)
        settings.setValue('model_path', state.model_path)
        settings.setValue('network_name', state.network_name)
        settings.setValue('preprocess', state.preprocess)
        settings.setValue('threshold', state.threshold)
        settings.setValue('image_dir', state.image_dir)
        settings.setValue('image_path', state.image_path)
        settings.setValue('video_path', state.video_path)
        settings.setValue('media_kind', state.media_kind)
        settings.setValue('media_index', state.media_index)
        settings.setValue('preview_mode', state.preview_mode)
        settings.setValue('fill_overlay', state.fill_overlay)
        if state.geometry is not None:
            settings.setValue('geometry', state.geometry)
        if state.window_state is not None:
            settings.setValue('window_state', state.window_state)
        settings.sync()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)
