# based on https://github.com/open-mmlab/mmdetection/blob/f78af7785ada87f1ced75a2313746e4ba3149760/mmdet/datasets/samplers/batch_sampler.py#L12  # noqa
from collections.abc import Generator

import numpy as np
from torch.utils.data import BatchSampler, Sampler

from diffengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a
    same batch.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
    """

    def __init__(self,
                 sampler: Sampler,
                 batch_size: int,
                 *,
                 drop_last: bool = False) -> None:
        if not isinstance(sampler, Sampler):
            msg = ("sampler should be an instance of ``Sampler``, but "
                   f"got {sampler}")
            raise TypeError(msg)
        if not isinstance(batch_size, int) or batch_size <= 0:
            msg = ("batch_size should be a positive integer value, but "
                   f"got batch_size={batch_size}")
            raise ValueError(msg)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        # two groups for w < h and w >= h
        self._aspect_ratio_buckets: dict = {}
        # calc aspect ratio
        self.bucket_ids = []
        for idx in range(len(self.sampler.dataset)):
            data_info = self.sampler.dataset[idx]
            bucket_id = data_info["inputs"]["img"].size(
            )[1] / data_info["inputs"]["img"].size()[2]
            self.bucket_ids.append(bucket_id)

    def __iter__(self) -> Generator:
        for idx in self.sampler:
            bucket_id = self.bucket_ids[idx]
            if bucket_id not in self._aspect_ratio_buckets:
                self._aspect_ratio_buckets[bucket_id] = []
            bucket = self._aspect_ratio_buckets[bucket_id]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the bucket
        if not self.drop_last:
            for v in self._aspect_ratio_buckets.values():
                if len(v) > 0:
                    yield v
                    del v
        self._aspect_ratio_buckets = {}

    def __len__(self) -> int:
        total_sample = 0
        _, counts = np.unique(self.bucket_ids, return_counts=True)
        for c in counts:
            if self.drop_last:
                total_sample += c // self.batch_size
            else:
                total_sample += c // self.batch_size
                if c % self.batch_size != 0:
                    total_sample += 1
        return total_sample
