from unittest import TestCase
from unittest.mock import patch

import torch
from mmengine.dataset import DefaultSampler
from torch.utils.data import Dataset

from diffengine.datasets.samplers import AspectRatioBatchSampler


class DummyDataset(Dataset):

    def __init__(self, length):
        self.length = length
        self.shapes = [[32, 32] if i % 2 == 0 else [16, 48]
                       for i in range(length)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        results = dict(
            img=torch.zeros((3, self.shapes[idx][0], self.shapes[idx][1])),
            aspect_ratio=self.shapes[idx][0] / self.shapes[idx][1])
        return dict(inputs=results)


class TestAspectRatioBatchSampler(TestCase):

    @patch('mmengine.dist.get_dist_info', return_value=(0, 1))
    def setUp(self, mock):
        self.length = 100
        self.dataset = DummyDataset(self.length)
        self.sampler = DefaultSampler(self.dataset, shuffle=False)

    def test_invalid_inputs(self):
        with self.assertRaisesRegex(
                ValueError, 'batch_size should be a positive integer value'):
            AspectRatioBatchSampler(self.sampler, batch_size=-1)

        with self.assertRaisesRegex(
                TypeError, 'sampler should be an instance of ``Sampler``'):
            AspectRatioBatchSampler(None, batch_size=1)

    def test_divisible_batch(self):
        batch_size = 5
        batch_sampler = AspectRatioBatchSampler(
            self.sampler, batch_size=batch_size, drop_last=True)
        self.assertEqual(
            len(batch_sampler), (self.length / 2 // batch_size) * 2)
        for batch_idxs in batch_sampler:
            self.assertEqual(len(batch_idxs), batch_size)
            batch = [
                self.dataset[idx]['inputs']['aspect_ratio']
                for idx in batch_idxs
            ]
            for i in range(1, batch_size):
                self.assertEqual(batch[0], batch[i])

    def test_indivisible_batch(self):
        batch_size = 7
        batch_sampler = AspectRatioBatchSampler(
            self.sampler, batch_size=batch_size, drop_last=True)
        all_batch_idxs = list(batch_sampler)
        self.assertEqual(
            len(batch_sampler), (self.length / 2 // batch_size) * 2)
        self.assertEqual(
            len(all_batch_idxs), (self.length / 2 // batch_size) * 2)

        batch_sampler = AspectRatioBatchSampler(
            self.sampler, batch_size=batch_size, drop_last=False)
        all_batch_idxs = list(batch_sampler)
        self.assertEqual(
            len(batch_sampler), (self.length / 2 // batch_size) * 2 + 2)
        self.assertEqual(
            len(all_batch_idxs), (self.length / 2 // batch_size) * 2 + 2)

        # the last batch may not have the same aspect ratio
        for batch_idxs in all_batch_idxs[:-2]:
            self.assertEqual(len(batch_idxs), batch_size)
            batch = [
                self.dataset[idx]['inputs']['aspect_ratio']
                for idx in batch_idxs
            ]
            for i in range(1, batch_size):
                self.assertEqual(batch[0], batch[i])
