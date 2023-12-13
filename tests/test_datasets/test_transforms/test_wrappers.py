import warnings
from unittest import TestCase

from diffengine.datasets.transforms.base import BaseTransform
from diffengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class AddToValue(BaseTransform):
    """Dummy transform to add a given addend to results['value']"""

    def __init__(self, addend=0) -> None:
        super().__init__()
        self.addend = addend

    def add(self, results, addend):
        augend = results["value"]

        if isinstance(augend, list):
            warnings.warn("value is a list", UserWarning)
        if isinstance(augend, dict):
            warnings.warn("value is a dict", UserWarning)

        def _add_to_value(augend, addend):
            if isinstance(augend, list):
                return [_add_to_value(v, addend) for v in augend]
            if isinstance(augend, dict):
                return {k: _add_to_value(v, addend) for k, v in augend.items()}
            return augend + addend

        results["value"] = _add_to_value(results["value"], addend)
        return results

    def transform(self, results):
        return self.add(results, self.addend)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"addend = {self.addend}"
        return repr_str


class TestRandomChoice(TestCase):

    def test_register(self):
        assert "RandomChoice" in TRANSFORMS

    def test_transform(self):
        data = dict(value=1)

        # test transform
        trans = TRANSFORMS.build(dict(type="RandomChoice",
                                      transforms=[
                                          [AddToValue(addend=1.0)],
                                          [AddToValue(addend=2.0)]],
                                      prob=[1.0, 0.0]))
        data = trans(data)
        assert data["value"] == 2

        # Case 2: default probability
        trans = TRANSFORMS.build(dict(type="RandomChoice",
                                      transforms=[
                                          [AddToValue(addend=1.0)],
                                          [AddToValue(addend=2.0)]]))

        _ = trans(dict(value=1))
