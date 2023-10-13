from unittest import TestCase

from diffengine.engine import TRANSFORMER_OPTIMIZERS


class TestBuilder(TestCase):

    def test_torch_optimizers(self):
        torch_optimizers = [
            "Adafactor",
        ]
        assert set(torch_optimizers).issubset(set(TRANSFORMER_OPTIMIZERS))
