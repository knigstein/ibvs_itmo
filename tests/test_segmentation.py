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


if __name__ == "__main__":
    test_brown_square_on_blue()
    print("test_segmentation OK")
