import copy
import unittest

import torch

from diffengine.registry import TRANSFORMS


class TestPackInputs(unittest.TestCase):

    def test_transform(self):
        data = {'dummy': 1, 'img': torch.zeros((3, 32, 32)), 'text': 'a'}

        cfg = dict(type='PackInputs', input_keys=['img', 'text'])
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        self.assertIn('inputs', results)

        self.assertIn('img', results['inputs'])
        self.assertIsInstance(results['img'], torch.Tensor)
        self.assertIn('text', results['inputs'])
        self.assertIsInstance(results['text'], str)
        self.assertNotIn('dummy', results['inputs'])
