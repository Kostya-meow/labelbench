"""Exact upstream mask NMS with cached areas and intersections restricted to their ROI."""

import numpy as np
from manuscript.detectors import Mask2Former


class MemoryEfficientMask2Former(Mask2Former):
    def _postprocess(self, class_logits: np.ndarray, mask_logits: np.ndarray, geometry: tuple) -> list:
        self._mask_metadata = {}
        try:
            return super()._postprocess(class_logits, mask_logits, geometry)
        finally:
            self._mask_metadata.clear()

    def _metadata(self, mask: np.ndarray) -> tuple:
        key = id(mask)
        if key not in self._mask_metadata:
            xs = np.flatnonzero(mask.any(axis=0))
            ys = np.flatnonzero(mask.any(axis=1))
            bounds = (int(xs[0]), int(ys[0]), int(xs[-1])+1, int(ys[-1])+1) if len(xs) else (0, 0, 0, 0)
            self._mask_metadata[key] = (int(mask.sum()), bounds)
        return self._mask_metadata[key]

    def _mask_iou(self, first: np.ndarray, second: np.ndarray) -> float:
        first_area, a = self._metadata(first)
        second_area, b = self._metadata(second)
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.
        intersection = int(np.count_nonzero(first[y1:y2, x1:x2] & second[y1:y2, x1:x2]))
        union = first_area+second_area-intersection
        return intersection / union if union else 0.
