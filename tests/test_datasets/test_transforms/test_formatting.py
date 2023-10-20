import copy
import unittest

import torch

from diffengine.registry import TRANSFORMS


class TestPackInputs(unittest.TestCase):

    def test_transform(self):
        data = {"dummy": 1, "img": torch.zeros((3, 32, 32)), "text": "a"}

        cfg = dict(type="PackInputs", input_keys=["img", "text"])
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        assert "inputs" in results

        assert "img" in results["inputs"]
        assert isinstance(results["inputs"]["img"], torch.Tensor)
        assert "text" in results["inputs"]
        assert isinstance(results["inputs"]["text"], str)
        assert "dummy" not in results["inputs"]
