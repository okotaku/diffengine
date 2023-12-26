from typing import Any
from unittest import TestCase
from unittest.mock import patch

import pytest
import torch
from mmengine.dataset import DefaultSampler
from torch.utils.data import Dataset

from diffengine.datasets.samplers import AspectRatioBatchSampler


class DummyDataset(Dataset):

    def __init__(self, length) -> None:
        self.length = length
        self.shapes = [[32, 32] if i % 2 == 0 else [16, 48]
                       for i in range(length)]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> dict[str, dict[str, Any]]:
        results = dict(
            img=torch.zeros((3, self.shapes[idx][0], self.shapes[idx][1])),
            aspect_ratio=self.shapes[idx][0] / self.shapes[idx][1])
        return dict(inputs=results)


class DummyDataset2(Dataset):

    def __init__(self) -> None:
        self.shapes = [[32, 32], [32, 32], [16, 48], [16, 48],
                       [32, 32], [16, 48], [32, 32]]

    def __len__(self) -> int:
        return len(self.shapes)

    def __getitem__(self, idx) -> dict[str, dict[str, Any]]:
        results = dict(
            img=torch.zeros((3, self.shapes[idx][0], self.shapes[idx][1])),
            aspect_ratio=self.shapes[idx][0] / self.shapes[idx][1])
        return dict(inputs=results)


class TestAspectRatioBatchSampler(TestCase):

    @patch("mmengine.dist.get_dist_info", return_value=(0, 1))
    def setUp(self, mock):  # noqa
        self.length = 100
        self.dataset = DummyDataset(self.length)
        self.sampler = DefaultSampler(self.dataset, shuffle=False)

    def test_invalid_inputs(self):
        with pytest.raises(
                ValueError,
                match="batch_size should be a positive integer value"):
            AspectRatioBatchSampler(self.sampler, batch_size=-1)

        with pytest.raises(
                TypeError,
                match="sampler should be an instance of ``Sampler``"):
            AspectRatioBatchSampler(None, batch_size=1)

    def test_divisible_batch(self):
        batch_size = 5
        batch_sampler = AspectRatioBatchSampler(
            self.sampler, batch_size=batch_size, drop_last=True)
        assert len(batch_sampler) == self.length / 2 // batch_size * 2
        for batch_idxs in batch_sampler:
            assert len(batch_idxs) == batch_size
            batch = [
                self.dataset[idx]["inputs"]["aspect_ratio"]
                for idx in batch_idxs
            ]
            for i in range(1, batch_size):
                assert batch[0] == batch[i]

    def test_indivisible_batch(self):
        batch_size = 7
        batch_sampler = AspectRatioBatchSampler(
            self.sampler, batch_size=batch_size, drop_last=True)
        all_batch_idxs = list(batch_sampler)
        assert len(batch_sampler) == self.length / 2 // batch_size * 2
        assert len(all_batch_idxs) == self.length / 2 // batch_size * 2

        batch_sampler = AspectRatioBatchSampler(
            self.sampler, batch_size=batch_size, drop_last=False)
        all_batch_idxs = list(batch_sampler)
        assert len(batch_sampler) == self.length / 2 // batch_size * 2 + 2
        assert len(all_batch_idxs) == self.length / 2 // batch_size * 2 + 2

        # the last batch may not have the same aspect ratio
        for batch_idxs in all_batch_idxs[:-2]:
            assert len(batch_idxs) == batch_size
            batch = [
                self.dataset[idx]["inputs"]["aspect_ratio"]
                for idx in batch_idxs
            ]
            for i in range(1, batch_size):
                assert batch[0] == batch[i]


class TestAspectRatioBatchSamplerFromfile(TestCase):

    @patch("mmengine.dist.get_dist_info", return_value=(0, 1))
    def setUp(self, mock):  # noqa
        self.dataset = DummyDataset2()
        self.sampler = DefaultSampler(self.dataset, shuffle=False)

    def test_invalid_inputs(self):
        with pytest.raises(
                ValueError,
                match="batch_size should be a positive integer value"):
            AspectRatioBatchSampler(self.sampler, batch_size=-1)

        with pytest.raises(
                TypeError,
                match="sampler should be an instance of ``Sampler``"):
            AspectRatioBatchSampler(None, batch_size=1)

    def test_divisible_batch(self):
        batch_size = 2
        batch_sampler = AspectRatioBatchSampler(
            self.sampler, batch_size=batch_size, drop_last=True,
            bucket_ids="tests/testdata/bucket_ids.pkl")
        assert len(batch_sampler) == len(self.dataset) // batch_size
        for batch_idxs in batch_sampler:
            assert len(batch_idxs) == batch_size
            batch = [
                self.dataset[idx]["inputs"]["aspect_ratio"]
                for idx in batch_idxs
            ]
            for i in range(1, batch_size):
                assert batch[0] == batch[i]

    def test_indivisible_batch(self):
        batch_size = 3
        batch_sampler = AspectRatioBatchSampler(
            self.sampler, batch_size=batch_size, drop_last=True,
            bucket_ids="tests/testdata/bucket_ids.pkl")
        all_batch_idxs = list(batch_sampler)
        assert len(batch_sampler) == len(self.dataset) // batch_size
        assert len(all_batch_idxs) == len(self.dataset) // batch_size

        batch_sampler = AspectRatioBatchSampler(
            self.sampler, batch_size=batch_size, drop_last=False,
            bucket_ids="tests/testdata/bucket_ids.pkl")
        all_batch_idxs = list(batch_sampler)
        assert len(batch_sampler) == len(self.dataset) // batch_size + 1
        assert len(all_batch_idxs) == len(self.dataset) // batch_size + 1

        # the last batch may not have the same aspect ratio
        for batch_idxs in all_batch_idxs[:-2]:
            assert len(batch_idxs) == batch_size
            batch = [
                self.dataset[idx]["inputs"]["aspect_ratio"]
                for idx in batch_idxs
            ]
            for i in range(1, batch_size):
                assert batch[0] == batch[i]
