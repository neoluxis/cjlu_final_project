"""ONNX Runtime inference wrapper."""

from __future__ import annotations

import time
from dataclasses import dataclass
from os import environ
from pathlib import Path

import cv2
import numpy as np

from constants import DEFAULT_INPUT_SIZE


@dataclass(frozen=True)
class InferenceOutput:
    logits: np.ndarray
    elapsed_ms: float


class OnnxSegmentor:
    def __init__(self) -> None:
        self.session = None
        self.model_path: Path | None = None
        self.input_name = 'input'
        self.output_name = 'seg_logits'
        self.input_size = DEFAULT_INPUT_SIZE

    def is_loaded(self) -> bool:
        return self.session is not None

    def provider_names(self) -> list[str]:
        if self.session is None:
            return []
        return list(self.session.get_providers())

    def load(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                'onnxruntime is not installed. Install it to run inference.'
            ) from exc

        path = Path(model_path)
        providers = _select_providers(ort.get_available_providers())
        if 'CUDAExecutionProvider' in providers and hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.model_path = path
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError('ONNX model must have at least one input and one output.')
        self.input_name = inputs[0].name
        self.output_name = (
            'seg_logits'
            if any(output.name == 'seg_logits' for output in outputs)
            else outputs[0].name)

        shape = inputs[0].shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_size = (int(shape[2]), int(shape[3]))
        else:
            self.input_size = DEFAULT_INPUT_SIZE

    def infer(self, image_rgb: np.ndarray) -> InferenceOutput:
        if self.session is None:
            raise RuntimeError('No ONNX model is loaded.')
        input_tensor = self._to_input_tensor(image_rgb)
        start = time.perf_counter()
        outputs = self.session.run(
            [self.output_name], {self.input_name: input_tensor})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logits = np.asarray(outputs[0])
        return InferenceOutput(logits=logits, elapsed_ms=elapsed_ms)

    def _to_input_tensor(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = self.input_size
        resized = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32).transpose(2, 0, 1)[None, ...]
        return tensor


def _select_providers(available_providers: list[str]) -> list[str]:
    available = set(available_providers)
    providers = ['CPUExecutionProvider']
    use_cuda = environ.get('SEGFORMER_APP_USE_CUDA', '').lower() in {
        '1', 'true', 'yes', 'on'}
    if use_cuda and 'CUDAExecutionProvider' in available:
        providers.insert(0, 'CUDAExecutionProvider')
    return [provider for provider in providers if provider in available] or [
        'CPUExecutionProvider']
