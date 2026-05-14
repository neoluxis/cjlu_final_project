"""Model discovery and filename-based metadata inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelEntry:
    path: Path
    display_name: str
    network_name: str
    recommended_preprocess: str


def identify_network_name(path: str | Path) -> str:
    name = str(path).lower()
    if 'segformer' in name or 'mit-b0' in name or 'mit_b0' in name:
        return 'SegFormer'
    if 'deeplabv3plus' in name or 'deeplabv3+' in name:
        return 'DeepLabV3+'
    if 'pspnet' in name:
        return 'PSPNet'
    if 'mask2former' in name:
        return 'Mask2Former'
    if 'unet' in name:
        return 'UNet'
    return 'Unknown Network'


def recommend_preprocess(path: str | Path) -> str:
    name = str(path).lower()
    if '_clahe' in name:
        return 'clahe'
    if '_gaussian' in name:
        return 'gaussian'
    if '_gamma' in name:
        return 'gamma'
    return 'none'


def scan_models(folder: str | Path) -> list[ModelEntry]:
    root = Path(folder).expanduser()
    if not root.is_dir():
        return []

    entries: list[ModelEntry] = []
    for path in sorted(root.rglob('*.onnx')):
        try:
            display_name = str(path.relative_to(root))
        except ValueError:
            display_name = path.name
        entries.append(
            ModelEntry(
                path=path,
                display_name=display_name,
                network_name=identify_network_name(path),
                recommended_preprocess=recommend_preprocess(path),
            ))
    return entries

