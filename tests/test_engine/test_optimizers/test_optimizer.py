import unittest
from unittest import TestCase

from diffengine.engine import APEX_OPTIMIZERS

try:
    import apex
except ImportError:
    apex = None


class TestBuilder(TestCase):

    @unittest.skipIf(apex is None, "apex is not installed")
    def test_apex_optimizers(self) -> None:
        apex_optimizers = [
            "FusedAdam", "FusedSGD",
        ]
        assert set(apex_optimizers).issubset(set(APEX_OPTIMIZERS))
