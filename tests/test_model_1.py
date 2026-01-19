import pytest
import torch
import numpy as np
from model_1.model_odlocanja import model

class DummyOBB:
    def __init__(self, xywhr, cls):
        self.xywhr = torch.tensor(xywhr)
        self.cls = torch.tensor(cls)

class DummyResult:
    def __init__(self, xywhr, cls, shape):
        self.obb = DummyOBB(xywhr, cls)
        self.orig_shape = shape  # (height, width)

@pytest.mark.parametrize("xywhr, expected", [
    ([[100, 100, 50, 20, 0]], 0),  # far left → not in central X zone
    ([[400, 200, 50, 50, 0]], 0),  # center X but Y too high
    ([[400, 490, 50, 50, 0]], 3),  # center X + Y very low → danger = 3
])
def test_izracunaj_ovire(xywhr, expected):
    shape = (500, 800)  # height, width
    dummy = DummyResult(xywhr, [0], shape)
    result = model.izracunaj_ovire(dummy)
    assert result == expected
