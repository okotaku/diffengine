from unittest import TestCase

import torch
from mmengine.optim import OptimWrapper
from torch.optim import SGD

from diffengine.models.editors import StableDiffusionXL
from diffengine.models.losses import L2Loss


class TestStableDiffusionXL(TestCase):

    def test_infer(self):
        StableDiffuser = StableDiffusionXL(
            'hf-internal-testing/tiny-stable-diffusion-xl-pipe')

        # test infer
        result = StableDiffuser.infer(
            ['an insect robot preparing a delicious meal'],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # test device
        assert StableDiffuser.device.type == 'cpu'

    def test_train_step(self):
        # test load with loss module
        StableDiffuser = StableDiffusionXL(
            'hf-internal-testing/tiny-stable-diffusion-xl-pipe', loss=L2Loss())

        # test train step
        data = dict(
            img=[torch.zeros((3, 64, 64))],
            text=['a dog'],
            time_ids=[torch.zeros((1, 6))])
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        self.assertIsInstance(log_vars['loss'], torch.Tensor)

        # test forward
        with self.assertRaisesRegex(NotImplementedError, 'Forward is not'):
            StableDiffuser.forward(torch.zeros((1, )))
