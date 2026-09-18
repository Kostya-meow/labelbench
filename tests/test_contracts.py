from pathlib import Path

import numpy as np
import pytest

from labelbench.contracts import safe_child_path
from labelbench.providers.mask2former import encode_binary_mask
from labelbench.service import AnnotationService


def test_safe_child_path_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        safe_child_path(tmp_path, "../escape.jpg")


def test_binary_mask_rle_covers_all_pixels() -> None:
    rle = encode_binary_mask(np.array([[False, True], [False, True]]))
    assert rle["size"] == [2, 2]
    assert sum(rle["counts"]) == 4


def test_iou() -> None:
    assert AnnotationService._iou([0, 0, 10, 10], [5, 5, 10, 10]) == pytest.approx(25 / 175)
