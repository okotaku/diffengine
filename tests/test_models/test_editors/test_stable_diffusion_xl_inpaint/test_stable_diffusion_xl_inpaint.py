from unittest import TestCase

import pytest
import torch
from mmengine.optim import OptimWrapper
from torch.optim import SGD

from diffengine.models.editors import (
    SDXLInpaintDataPreprocessor,
    StableDiffusionXLInpaint,
)
from diffengine.models.losses import L2Loss


class TestStableDiffusionXL(TestCase):

    def test_init(self):
        with pytest.raises(
                AssertionError, match="If you want to use LoRA"):
            _ = StableDiffusionXLInpaint(
                model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
                text_encoder_lora_config=dict(type="dummy"),
                data_preprocessor=SDXLInpaintDataPreprocessor())

        with pytest.raises(
                AssertionError, match="If you want to finetune text"):
            _ = StableDiffusionXLInpaint(
                model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
                unet_lora_config=dict(type="dummy"),
                finetune_text_encoder=True,
                data_preprocessor=SDXLInpaintDataPreprocessor())

    def test_infer(self):
        StableDiffuser = StableDiffusionXLInpaint(
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            data_preprocessor=SDXLInpaintDataPreprocessor())

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            ["tests/testdata/color.jpg"],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # test device
        assert StableDiffuser.device.type == "cpu"

        # test infer with negative_prompt
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            ["tests/testdata/color.jpg"],
            negative_prompt="noise",
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # output_type = 'latent'
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            ["tests/testdata/color.jpg"],
            output_type="latent",
            height=64,
            width=64)
        assert len(result) == 1
        assert type(result[0]) == torch.Tensor
        assert result[0].shape == (4, 32, 32)

    def test_infer_v_prediction(self):
        StableDiffuser = StableDiffusionXLInpaint(
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            data_preprocessor=SDXLInpaintDataPreprocessor(),
            prediction_type="v_prediction")
        assert StableDiffuser.prediction_type == "v_prediction"

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            ["tests/testdata/color.jpg"],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

    def test_infer_with_lora(self):
        StableDiffuser = StableDiffusionXLInpaint(
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            unet_lora_config=dict(
                type="LoRA", r=4,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config = dict(
                type="LoRA", r=4,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
            data_preprocessor=SDXLInpaintDataPreprocessor())

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            ["tests/testdata/color.jpg"],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

    def test_infer_with_pre_compute_embs(self):
        StableDiffuser = StableDiffusionXLInpaint(
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            pre_compute_text_embeddings=True,
            data_preprocessor=SDXLInpaintDataPreprocessor())

        assert not hasattr(StableDiffuser, "tokenizer_one")
        assert not hasattr(StableDiffuser, "text_encoder_one")
        assert not hasattr(StableDiffuser, "tokenizer_two")
        assert not hasattr(StableDiffuser, "text_encoder_two")

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            ["tests/testdata/color.jpg"],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # test device
        assert StableDiffuser.device.type == "cpu"

    def test_train_step(self):
        # test load with loss module
        StableDiffuser = StableDiffusionXLInpaint(
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            loss=L2Loss(),
            data_preprocessor=SDXLInpaintDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                mask=[torch.zeros((1, 64, 64))],
                masked_image=[torch.zeros((3, 64, 64))],
                text=["a dog"],
                time_ids=[torch.zeros((1, 6))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_lora(self):
        # test load with loss module
        StableDiffuser = StableDiffusionXLInpaint(
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            loss=L2Loss(),
            unet_lora_config=dict(
                type="LoRA", r=4,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config = dict(
                type="LoRA", r=4,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
            data_preprocessor=SDXLInpaintDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                mask=[torch.zeros((1, 64, 64))],
                masked_image=[torch.zeros((3, 64, 64))],
                text=["a dog"],
                time_ids=[torch.zeros((1, 6))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_input_perturbation(self):
        # test load with loss module
        StableDiffuser = StableDiffusionXLInpaint(
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            input_perturbation_gamma=0.1,
            loss=L2Loss(),
            data_preprocessor=SDXLInpaintDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                mask=[torch.zeros((1, 64, 64))],
                masked_image=[torch.zeros((3, 64, 64))],
                text=["a dog"],
                time_ids=[torch.zeros((1, 6))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_gradient_checkpointing(self):
        # test load with loss module
        StableDiffuser = StableDiffusionXLInpaint(
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            loss=L2Loss(),
            data_preprocessor=SDXLInpaintDataPreprocessor(),
            gradient_checkpointing=True)

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                mask=[torch.zeros((1, 64, 64))],
                masked_image=[torch.zeros((3, 64, 64))],
                text=["a dog"],
                time_ids=[torch.zeros((1, 6))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_pre_compute_embs(self):
        # test load with loss module
        StableDiffuser = StableDiffusionXLInpaint(
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            pre_compute_text_embeddings=True,
            loss=L2Loss(),
            data_preprocessor=SDXLInpaintDataPreprocessor())

        assert not hasattr(StableDiffuser, "tokenizer_one")
        assert not hasattr(StableDiffuser, "text_encoder_one")
        assert not hasattr(StableDiffuser, "tokenizer_two")
        assert not hasattr(StableDiffuser, "text_encoder_two")

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                mask=[torch.zeros((1, 64, 64))],
                masked_image=[torch.zeros((3, 64, 64))],
                prompt_embeds=[torch.zeros((77, 64))],
                pooled_prompt_embeds=[torch.zeros(32)],
                time_ids=[torch.zeros((1, 6))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_dreambooth(self):
        # test load with loss module
        StableDiffuser = StableDiffusionXLInpaint(
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            loss=L2Loss(),
            data_preprocessor=SDXLInpaintDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                mask=[torch.zeros((1, 64, 64))],
                masked_image=[torch.zeros((3, 64, 64))],
                text=["a sks dog"],
                time_ids=[torch.zeros((1, 6))]))
        data["inputs"]["result_class_image"] = dict(
            img=[torch.zeros((3, 64, 64))],
            mask=[torch.zeros((1, 64, 64))],
            masked_image=[torch.zeros((3, 64, 64))],
            text=["a dog"],
            time_ids=[torch.zeros((1, 6))])  # type: ignore[assignment]
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_v_prediction(self):
        # test load with loss module
        StableDiffuser = StableDiffusionXLInpaint(
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            loss=L2Loss(),
            prediction_type="v_prediction",
            data_preprocessor=SDXLInpaintDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                mask=[torch.zeros((1, 64, 64))],
                masked_image=[torch.zeros((3, 64, 64))],
                text=["a dog"],
                time_ids=[torch.zeros((1, 6))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_val_and_test_step(self):
        StableDiffuser = StableDiffusionXLInpaint(
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            loss=L2Loss(),
            data_preprocessor=SDXLInpaintDataPreprocessor())

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            StableDiffuser.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            StableDiffuser.test_step(torch.zeros((1, )))