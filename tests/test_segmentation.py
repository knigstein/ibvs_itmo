import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision import CubeSegmenter  # noqa: E402


def test_brown_square_on_blue():
    img = np.full((480, 640, 3), (120, 90, 60), dtype=np.uint8)
    cv2.rectangle(img, (200, 140), (440, 340), (60, 120, 180), -1)
    seg = CubeSegmenter(
        {
            "hsv_lower": [5, 40, 40],
            "hsv_upper": [40, 255, 255],
            "min_area": 200,
        }
    )
    r = seg.detect(img)
    assert r.ok, r.meta
    assert r.corners.shape == (4, 2)


def test_empty_image_not_ok():
    r = CubeSegmenter({}).detect(np.zeros((0, 0, 3), dtype=np.uint8))
    assert not r.ok


def test_uniform_background_no_contour():
    img = np.full((480, 640, 3), (200, 200, 200), dtype=np.uint8)
    r = CubeSegmenter({"min_area": 100}).detect(img)
    assert not r.ok


def test_min_area_rejects_small_blob():
    img = np.full((480, 640, 3), (120, 90, 60), dtype=np.uint8)
    cv2.circle(img, (320, 240), 8, (60, 120, 180), -1)
    r = CubeSegmenter({"hsv_lower": [5, 40, 40], "hsv_upper": [40, 255, 255], "min_area": 5000}).detect(
        img
    )
    assert not r.ok


def test_reset_smoothing_clears_state():
    seg = CubeSegmenter({"ema_alpha": 0.5})
    img = np.full((480, 640, 3), (120, 90, 60), dtype=np.uint8)
    cv2.rectangle(img, (200, 140), (440, 340), (60, 120, 180), -1)
    seg.detect(img)
    assert seg._corners_ema is not None
    seg.reset_smoothing()
    assert seg._corners_ema is None


if __name__ == "__main__":
    test_brown_square_on_blue()
    test_empty_image_not_ok()
    test_uniform_background_no_contour()
    test_min_area_rejects_small_blob()
    test_reset_smoothing_clears_state()
    print("test_segmentation OK")
