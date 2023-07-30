from collections.abc import Sequence
from typing import List

import numpy as np
import torch
from mmengine.utils import is_str

from diffengine.datasets.transforms import BaseTransform
from diffengine.registry import TRANSFORMS


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


@TRANSFORMS.register_module()
class PackInputs(BaseTransform):
    """Pack the inputs data.

    **Required Keys:**

    - ``input_key``

    **Deleted Keys:**

    All other keys in the dict.

    Args:
        input_keys (List[str]): The key of element to feed into the model
            forwarding. Defaults to ['img', 'text'].
        skip_to_tensor_key (List[str]): The key of element to skip to_tensor.
            Defaults to ['text'].
    """

    def __init__(self,
                 input_keys: List[str] = ['img', 'text'],
                 skip_to_tensor_key: List[str] = ['text']):
        self.input_keys = input_keys
        self.skip_to_tensor_key = skip_to_tensor_key

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""

        packed_results = dict()
        for k in self.input_keys:
            if k in results and k not in self.skip_to_tensor_key:
                packed_results[k] = to_tensor(results[k])
            elif k in results:
                # text skip to_tensor
                packed_results[k] = results[k]

        return packed_results
