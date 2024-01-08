from unittest import TestCase

import pytest
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from mmengine.optim import OptimWrapper
from torch.optim import SGD
from transformers import CLIPTextModel, CLIPTokenizer

from diffengine.models.editors import SDInpaintDataPreprocessor, StableDiffusionInpaint
from diffengine.models.losses import DeBiasEstimationLoss, L2Loss, SNRL2Loss
from diffengine.registry import MODELS


class TestStableDiffusionInpaint(TestCase):

    def _get_config(self) -> dict:
        base_model = "diffusers/tiny-stable-diffusion-torch"
        return dict(
            type=StableDiffusionInpaint,
             model=base_model,
             tokenizer=dict(type=CLIPTokenizer.from_pretrained,
                            pretrained_model_name_or_path=base_model,
                            subfolder="tokenizer"),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            pretrained_model_name_or_path=base_model,
                            subfolder="scheduler"),
             text_encoder=dict(type=CLIPTextModel.from_pretrained,
                               pretrained_model_name_or_path=base_model,
                               subfolder="text_encoder"),
             vae=dict(
                type=AutoencoderKL.from_pretrained,
                pretrained_model_name_or_path=base_model,
                subfolder="vae"),
             unet=dict(type=UNet2DConditionModel.from_pretrained,
                             pretrained_model_name_or_path=base_model,
                             subfolder="unet"),
            data_preprocessor=dict(type=SDInpaintDataPreprocessor),
            loss=dict(type=L2Loss))

    def test_init(self):
        cfg = self._get_config()
        cfg.update(text_encoder_lora_config=dict(type="dummy"))
        with pytest.raises(
                AssertionError, match="If you want to use LoRA"):
            _ = MODELS.build(cfg)

        cfg = self._get_config()
        cfg.update(unet_lora_config=dict(type="dummy"),
                finetune_text_encoder=True)
        with pytest.raises(
                AssertionError, match="If you want to finetune text"):
            _ = MODELS.build(cfg)

    def test_infer(self):
        cfg = self._get_config()
        StableDiffuser = MODELS.build(cfg)

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

    def test_infer_with_lora(self):
        cfg = self._get_config()
        cfg.update(
            unet_lora_config=dict(
                type="LoRA", r=4,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config = dict(
                type="LoRA", r=4,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
            finetune_text_encoder=True,
        )
        StableDiffuser = MODELS.build(cfg)

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            ["tests/testdata/color.jpg"],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

    def test_train_step(self):
        # test load with loss module
        cfg = self._get_config()
        StableDiffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))],
                        mask=[torch.zeros((1, 64, 64))],
                        masked_image=[torch.zeros((3, 64, 64))],
                        text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_v_prediction(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(prediction_type="v_prediction")
        StableDiffuser = MODELS.build(cfg)
        assert StableDiffuser.prediction_type == "v_prediction"

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))],
                        mask=[torch.zeros((1, 64, 64))],
                        masked_image=[torch.zeros((3, 64, 64))],
                        text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_lora(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(
            unet_lora_config=dict(
                type="LoRA", r=4,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config=dict(
                type="LoRA", r=4,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
        )
        StableDiffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))],
                        mask=[torch.zeros((1, 64, 64))],
                        masked_image=[torch.zeros((3, 64, 64))],
                        text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_input_perturbation(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(input_perturbation_gamma=0.1)
        StableDiffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))],
                        mask=[torch.zeros((1, 64, 64))],
                        masked_image=[torch.zeros((3, 64, 64))],
                        text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_gradient_checkpointing(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(gradient_checkpointing=True)
        StableDiffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))],
                        mask=[torch.zeros((1, 64, 64))],
                        masked_image=[torch.zeros((3, 64, 64))],
                        text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_dreambooth(self):
        # test load with loss module
        cfg = self._get_config()
        StableDiffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))],
                        mask=[torch.zeros((1, 64, 64))],
                        masked_image=[torch.zeros((3, 64, 64))],
                        text=["a sks dog"]))
        data["inputs"]["result_class_image"] = dict(
            img=[torch.zeros((3, 64, 64))],
            mask=[torch.zeros((1, 64, 64))],
            masked_image=[torch.zeros((3, 64, 64))],
            text=["a dog"])  # type: ignore[assignment]
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_snr_loss(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(loss=dict(type=SNRL2Loss))
        StableDiffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))],
                        mask=[torch.zeros((1, 64, 64))],
                        masked_image=[torch.zeros((3, 64, 64))],
                        text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_debias_estimation_loss(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(loss=dict(type=DeBiasEstimationLoss))
        StableDiffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))],
                        mask=[torch.zeros((1, 64, 64))],
                        masked_image=[torch.zeros((3, 64, 64))],
                        text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_val_and_test_step(self):
        cfg = self._get_config()
        StableDiffuser = MODELS.build(cfg)

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            StableDiffuser.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            StableDiffuser.test_step(torch.zeros((1, )))
