"""Prediction post-processing and overlay rendering."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from constants import SWRD_CLASSES, SWRD_PALETTE


@dataclass(frozen=True)
class ComponentInfo:
    component_id: int
    class_id: int
    class_name: str
    area: int
    area_ratio: float
    mean_confidence: float
    bbox: tuple[int, int, int, int]
    contours: tuple[np.ndarray, ...]


@dataclass
class PredictionResult:
    class_mask: np.ndarray
    confidence: np.ndarray
    instance_map: np.ndarray
    components: list[ComponentInfo]
    visible_class_ids: set[int]
    postprocess_ms: float

    def component_at(self, x: int, y: int) -> ComponentInfo | None:
        if y < 0 or x < 0 or y >= self.instance_map.shape[0] or x >= self.instance_map.shape[1]:
            return None
        component_id = int(self.instance_map[y, x])
        if component_id <= 0:
            return None
        for component in self.components:
            if component.component_id == component_id:
                return component
        return None

    def predicted_classes(self) -> list[int]:
        return sorted({component.class_id for component in self.components})


def postprocess_prediction(
    logits: np.ndarray,
    target_size: tuple[int, int],
    threshold: float,
) -> PredictionResult:
    start = time.perf_counter()
    if logits.ndim == 4:
        logits = logits[0]
    if logits.ndim != 3:
        raise ValueError(f'Expected logits [C,H,W], got shape {logits.shape}')

    probs = _softmax(logits, axis=0)
    confidence = np.max(probs, axis=0).astype(np.float32)
    class_mask = np.argmax(probs, axis=0).astype(np.uint8)

    target_h, target_w = target_size
    if class_mask.shape != (target_h, target_w):
        class_mask = cv2.resize(
            class_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        confidence = cv2.resize(
            confidence, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    class_mask = class_mask.copy()
    class_mask[confidence < threshold] = 0
    instance_map = np.zeros((target_h, target_w), dtype=np.int32)
    components: list[ComponentInfo] = []
    next_component_id = 1
    total_pixels = max(target_h * target_w, 1)

    for class_id in range(1, min(len(SWRD_CLASSES), int(class_mask.max()) + 1)):
        binary = (class_mask == class_id).astype(np.uint8)
        if binary.max() == 0:
            continue
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8)
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            component_mask = (labels == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            instance_map[labels == label_id] = next_component_id
            mean_conf = float(confidence[labels == label_id].mean())
            components.append(
                ComponentInfo(
                    component_id=next_component_id,
                    class_id=class_id,
                    class_name=SWRD_CLASSES[class_id],
                    area=area,
                    area_ratio=area / total_pixels,
                    mean_confidence=mean_conf,
                    bbox=(x, y, w, h),
                    contours=tuple(contours),
                ))
            next_component_id += 1

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PredictionResult(
        class_mask=class_mask,
        confidence=confidence,
        instance_map=instance_map,
        components=components,
        visible_class_ids=set(sorted({c.class_id for c in components})),
        postprocess_ms=elapsed_ms,
    )


def render_overlay(base_rgb: np.ndarray,
                   prediction: PredictionResult | None,
                   fill_enabled: bool = True) -> np.ndarray:
    image = base_rgb.copy()
    if prediction is None:
        return image

    for component in prediction.components:
        if component.class_id not in prediction.visible_class_ids:
            continue
        color = SWRD_PALETTE[component.class_id]
        if fill_enabled:
            filled = image.copy()
            cv2.drawContours(filled, list(component.contours), -1, color, -1)
            image = cv2.addWeighted(filled, 0.30, image, 0.70, 0)
        cv2.drawContours(image, list(component.contours), -1, color, 2)
    return image


def class_summary(prediction: PredictionResult) -> list[dict[str, object]]:
    rows = []
    for class_id in prediction.predicted_classes():
        components = [
            component for component in prediction.components
            if component.class_id == class_id
        ]
        total_area = sum(component.area for component in components)
        mean_conf = (
            sum(component.mean_confidence * component.area
                for component in components) / total_area
            if total_area else 0.0)
        rows.append({
            'class_id': class_id,
            'class_name': SWRD_CLASSES[class_id],
            'count': len(components),
            'area': total_area,
            'mean_confidence': mean_conf,
            'color': SWRD_PALETTE[class_id],
        })
    return rows


def tooltip_for_component(component: ComponentInfo) -> str:
    x, y, w, h = component.bbox
    return (
        f'Class: {component.class_name}\n'
        f'Confidence: {component.mean_confidence:.3f}\n'
        f'Area: {component.area} px ({component.area_ratio:.2%})\n'
        f'BBox: x={x}, y={y}, w={w}, h={h}\n'
        f'Region: {component.component_id}')


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(values)
    return exp / np.sum(exp, axis=axis, keepdims=True)
