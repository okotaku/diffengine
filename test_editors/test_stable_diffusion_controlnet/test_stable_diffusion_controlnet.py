from unittest import TestCase

import torch
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, DownBlock2D
from mmengine.optim import OptimWrapper
from torch.optim import SGD

from diffengine.models.editors import (SDControlNetDataPreprocessor,
                                       StableDiffusionControlNet)
from diffengine.models.losses import L2Loss


class TestStableDiffusionControlNet(TestCase):

    def test_init(self):
        with self.assertRaisesRegex(AssertionError,
                                    '`lora_config` should be None'):
            _ = StableDiffusionControlNet(
                'hf-internal-testing/tiny-stable-diffusion-pipe',
                controlnet_model='hf-internal-testing/tiny-controlnet',
                data_preprocessor=SDControlNetDataPreprocessor(),
                lora_config=dict(rank=4))

        with self.assertRaisesRegex(AssertionError,
                                    '`finetune_text_encoder` should be False'):
            _ = StableDiffusionControlNet(
                'hf-internal-testing/tiny-stable-diffusion-pipe',
                controlnet_model='hf-internal-testing/tiny-controlnet',
                data_preprocessor=SDControlNetDataPreprocessor(),
                finetune_text_encoder=True)

    def test_infer(self):
        StableDiffuser = StableDiffusionControlNet(
            'hf-internal-testing/tiny-stable-diffusion-pipe',
            controlnet_model='hf-internal-testing/tiny-controlnet',
            data_preprocessor=SDControlNetDataPreprocessor())
        assert isinstance(StableDiffuser.controlnet.down_blocks[1],
                          CrossAttnDownBlock2D)

        # test infer
        result = StableDiffuser.infer(
            ['an insect robot preparing a delicious meal'],
            ['tests/testdata/color.jpg'],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # test device
        assert StableDiffuser.device.type == 'cpu'

        # test infer with negative_prompt
        result = StableDiffuser.infer(
            ['an insect robot preparing a delicious meal'],
            ['tests/testdata/color.jpg'],
            negative_prompt='noise',
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # output_type = 'latent'
        result = StableDiffuser.infer(
            ['an insect robot preparing a delicious meal'],
            ['tests/testdata/color.jpg'],
            output_type='latent',
            height=64,
            width=64)
        assert len(result) == 1
        self.assertEqual(type(result[0]), torch.Tensor)
        self.assertEqual(result[0].shape, (4, 32, 32))

        # test controlnet small
        StableDiffuser = StableDiffusionControlNet(
            'hf-internal-testing/tiny-stable-diffusion-pipe',
            controlnet_model='hf-internal-testing/tiny-controlnet',
            data_preprocessor=SDControlNetDataPreprocessor(),
            transformer_layers_per_block=[0, 0])
        assert isinstance(StableDiffuser.controlnet.down_blocks[1],
                          DownBlock2D)

        result = StableDiffuser.infer(
            ['an insect robot preparing a delicious meal'],
            ['tests/testdata/color.jpg'],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

    def test_train_step(self):
        # test load with loss module
        StableDiffuser = StableDiffusionControlNet(
            'hf-internal-testing/tiny-stable-diffusion-pipe',
            controlnet_model='hf-internal-testing/tiny-controlnet',
            loss=L2Loss(),
            data_preprocessor=SDControlNetDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                text=['a dog'],
                condition_img=[torch.zeros((3, 64, 64))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        self.assertIsInstance(log_vars['loss'], torch.Tensor)

        # test controlnet small
        StableDiffuser = StableDiffusionControlNet(
            'hf-internal-testing/tiny-stable-diffusion-pipe',
            controlnet_model='hf-internal-testing/tiny-controlnet',
            data_preprocessor=SDControlNetDataPreprocessor(),
            transformer_layers_per_block=[0, 0])
        assert isinstance(StableDiffuser.controlnet.down_blocks[1],
                          DownBlock2D)

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                text=['a dog'],
                condition_img=[torch.zeros((3, 64, 64))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        self.assertIsInstance(log_vars['loss'], torch.Tensor)

    def test_train_step_with_gradient_checkpointing(self):
        # test load with loss module
        StableDiffuser = StableDiffusionControlNet(
            'hf-internal-testing/tiny-stable-diffusion-pipe',
            controlnet_model='hf-internal-testing/tiny-controlnet',
            loss=L2Loss(),
            data_preprocessor=SDControlNetDataPreprocessor(),
            gradient_checkpointing=True)

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                text=['a dog'],
                condition_img=[torch.zeros((3, 64, 64))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        self.assertIsInstance(log_vars['loss'], torch.Tensor)

    def test_val_and_test_step(self):
        StableDiffuser = StableDiffusionControlNet(
            'hf-internal-testing/tiny-stable-diffusion-pipe',
            controlnet_model='hf-internal-testing/tiny-controlnet',
            loss=L2Loss(),
            data_preprocessor=SDControlNetDataPreprocessor())

        # test val_step
        with self.assertRaisesRegex(NotImplementedError, 'val_step is not'):
            StableDiffuser.val_step(torch.zeros((1, )))

        # test test_step
        with self.assertRaisesRegex(NotImplementedError, 'test_step is not'):
            StableDiffuser.test_step(torch.zeros((1, )))
