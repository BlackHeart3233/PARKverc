import numpy as np
import pytest
import cv2

from model_1.augmentacija import clip, flip_polygon, transform

def test_clip():
    assert clip(1.5) == 1.0
    assert clip(-0.2) == 0.0
    assert clip(0.75) == 0.75

def test_flip_polygon():
    polygon = [0.1, 0.5, 0.3, 0.5, 0.3, 0.7, 0.1, 0.7]
    flipped = flip_polygon(polygon)
    expected = [0.9, 0.5, 0.7, 0.5, 0.7, 0.7, 0.9, 0.7]
    assert all(abs(a - b) < 1e-6 for a, b in zip(flipped, expected))

def test_transform_pipeline():
    # create dummy image (black 100x100)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # dummy label: 1 object at center
    labels = [(3, 0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.4, 0.6)]

    transformed = transform(img, labels)
    assert "image" in transformed
    assert "labels" in transformed
    assert isinstance(transformed["image"], np.ndarray)
    assert isinstance(transformed["labels"], list)
    assert all(len(lab) == 9 for lab in transformed["labels"])  # cls + 8 coords
