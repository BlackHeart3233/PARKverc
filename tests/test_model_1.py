import pytest
import numpy as np
from model_1.model_odlocanja import model  # adjust import path to your actual file

class DummyOBB:
    def __init__(self, xywhr, cls):
        self.xywhr = np.array(xywhr)
        self.cls = np.array(cls)

class DummyResult:
    def __init__(self, xywhr, cls, shape):
        self.obb = DummyOBB(xywhr, cls)
        self.orig_shape = shape  # (height, width)

@pytest.mark.parametrize("xywhr, expected", [
    # Not in center
    ([[100, 100, 50, 20, 0]], 1),  # far left
    # Center horizontally
    ([[400, 200, 50, 50, 0]], 2),  # x = 400 (in 800x width, middle)
    # Center + bottom of screen
    ([[400, 490, 50, 50, 0]], 3)   # y = 490 (bottom of 500px image)
])
def test_izracunaj_ovire(xywhr, expected):
    shape = (500, 800)  # height, width
    dummy = DummyResult(xywhr, [0], shape)
    result = model.izracunaj_ovire(dummy)
    assert result == expected
