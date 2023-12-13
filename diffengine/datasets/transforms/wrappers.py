from collections.abc import Callable, Iterator

import mmengine
import numpy as np
from mmengine.dataset.base_dataset import Compose

from diffengine.datasets.transforms.base import BaseTransform
from diffengine.registry import TRANSFORMS

Transform = dict | Callable[[dict], dict]


@TRANSFORMS.register_module()
class RandomChoice(BaseTransform):
    """Process data with a randomly chosen transform from given candidates.

    Copied from mmcv/transforms/wrappers.py.

    Args:
    ----
        transforms (list[list]): A list of transform candidates, each is a
            sequence of transforms.
        prob (list[float], optional): The probabilities associated
            with each pipeline. The length should be equal to the pipeline
            number and the sum should be 1. If not given, a uniform
            distribution will be assumed.

    Examples:
    --------
        >>> # config
        >>> pipeline = [
        >>>     dict(type='RandomChoice',
        >>>         transforms=[
        >>>             [dict(type='RandomHorizontalFlip')],  # subpipeline 1
        >>>             [dict(type='RandomRotate')],  # subpipeline 2
        >>>         ]
        >>>     )
        >>> ]
    """

    def __init__(self,
                 transforms: list[Transform | list[Transform]],
                 prob: list[float] | None = None) -> None:

        super().__init__()

        if prob is not None:
            assert mmengine.is_seq_of(prob, float)
            assert len(transforms) == len(prob),(
                "``transforms`` and ``prob`` must have same lengths. "
                f"Got {len(transforms)} vs {len(prob)}.")
            assert sum(prob) == 1

        self.prob = prob
        self.transforms = [Compose(transforms) for transforms in transforms]

    def __iter__(self) -> Iterator:
        """Iterate over transforms."""
        return iter(self.transforms)

    def random_pipeline_index(self) -> int:
        """Return a random transform index."""
        indices = np.arange(len(self.transforms))
        return np.random.choice(indices, p=self.prob)  # noqa

    def transform(self, results: dict) -> dict | None:
        """Randomly choose a transform to apply."""
        idx = self.random_pipeline_index()
        return self.transforms[idx](results)
