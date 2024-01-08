from unittest import TestCase

import pytest
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from mmengine.optim import OptimWrapper
from torch.optim import SGD
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from diffengine.models.editors import ESDXL, ESDXLDataPreprocessor
from diffengine.models.losses import L2Loss
from diffengine.registry import MODELS


class TestESDXL(TestCase):

    def _get_config(self) -> dict:
        base_model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        return dict(type=ESDXL,
             model=base_model,
             tokenizer_one=dict(type=AutoTokenizer.from_pretrained,
                            pretrained_model_name_or_path=base_model,
                            subfolder="tokenizer",
                            use_fast=False),
             tokenizer_two=dict(type=AutoTokenizer.from_pretrained,
                            pretrained_model_name_or_path=base_model,
                            subfolder="tokenizer_2",
                            use_fast=False),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            pretrained_model_name_or_path=base_model,
                            subfolder="scheduler"),
             text_encoder_one=dict(type=CLIPTextModel.from_pretrained,
                               pretrained_model_name_or_path=base_model,
                               subfolder="text_encoder"),
             text_encoder_two=dict(type=CLIPTextModelWithProjection.from_pretrained,
                               pretrained_model_name_or_path=base_model,
                               subfolder="text_encoder_2"),
             vae=dict(
                type=AutoencoderKL.from_pretrained,
                pretrained_model_name_or_path=base_model,
                subfolder="vae"),
             unet=dict(type=UNet2DConditionModel.from_pretrained,
                             pretrained_model_name_or_path=base_model,
                             subfolder="unet"),
            data_preprocessor=dict(type=ESDXLDataPreprocessor),
            negative_guidance=1.0,
            train_method="xattn",
            loss=dict(type=L2Loss))

    def test_init(self):
        cfg = self._get_config()
        cfg.update(pre_compute_text_embeddings=False)
        with pytest.raises(
                AssertionError,
                match="`pre_compute_text_embeddings` should be True"):
            _ = MODELS.build(cfg)

        cfg = self._get_config()
        cfg.update(finetune_text_encoder=True)
        with pytest.raises(
                AssertionError,
                match="`finetune_text_encoder` should be False"):
            _ = MODELS.build(cfg)

        cfg = self._get_config()
        cfg.update(prediction_type="v_prediction")
        with pytest.raises(
                AssertionError, match="`prediction_type` should be None"):
            _ = MODELS.build(cfg)

    def test_infer(self):
        cfg = self._get_config()
        cfg.update(
            unet_lora_config=dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
        )
        StableDiffuser = MODELS.build(cfg)

        assert not hasattr(StableDiffuser, "tokenizer_one")
        assert not hasattr(StableDiffuser, "text_encoder_one")
        assert not hasattr(StableDiffuser, "tokenizer_two")
        assert not hasattr(StableDiffuser, "text_encoder_two")

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # test device
        assert StableDiffuser.device.type == "cpu"

    def test_infer_with_lora(self):
        cfg = self._get_config()
        StableDiffuser = MODELS.build(cfg)

        assert not hasattr(StableDiffuser, "tokenizer_one")
        assert not hasattr(StableDiffuser, "text_encoder_one")
        assert not hasattr(StableDiffuser, "tokenizer_two")
        assert not hasattr(StableDiffuser, "text_encoder_two")

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # test device
        assert StableDiffuser.device.type == "cpu"

    def test_train_step_with_pre_compute_embs(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(train_method="xattn",
            height=64,
            width=64)
        StableDiffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(
                text=["dog"],
                prompt_embeds=[torch.zeros((2, 64))],
                pooled_prompt_embeds=[torch.zeros(32)],
                null_prompt_embeds=[torch.zeros((2, 64))],
                null_pooled_prompt_embeds=[torch.zeros(32)]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_lora(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(train_method="xattn",
            height=64,
            width=64,
            unet_lora_config=dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"]))
        StableDiffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(
                text=["dog"],
                prompt_embeds=[torch.zeros((2, 64))],
                pooled_prompt_embeds=[torch.zeros(32)],
                null_prompt_embeds=[torch.zeros((2, 64))],
                null_pooled_prompt_embeds=[torch.zeros(32)]))
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
