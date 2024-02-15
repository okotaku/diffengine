from unittest import TestCase

import pytest
import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.models.embeddings import IPAdapterPlusImageProjection
from mmengine.optim import OptimWrapper
from torch.optim import SGD
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from diffengine.models.editors import IPAdapterXLDataPreprocessor, IPAdapterXLPlus
from diffengine.models.losses import L2Loss
from diffengine.registry import MODELS


class TestIPAdapterXL(TestCase):

    def _get_config(self) -> dict:
        base_model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        return dict(type=IPAdapterXLPlus,
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
            image_encoder=dict(type=CLIPVisionModelWithProjection.from_pretrained,
                                pretrained_model_name_or_path="hf-internal-testing/unidiffuser-diffusers-test",
                                subfolder="image_encoder"),
            image_projection=dict(type=IPAdapterPlusImageProjection,
                                    hidden_dims=1280,
                                    depth=4,
                                    dim_head=64,
                                    heads=20,
                                    num_queries=16,
                                    ffn_ratio=4),
            feature_extractor=dict(
                type=CLIPImageProcessor.from_pretrained,
                pretrained_model_name_or_path="hf-internal-testing/unidiffuser-diffusers-test",
                subfolder="image_processor"),
            data_preprocessor=dict(type=IPAdapterXLDataPreprocessor),
            loss=dict(type=L2Loss))

    def test_init(self):
        cfg = self._get_config()
        cfg.update(unet_lora_config=dict(type="dummy"))
        with pytest.raises(
                AssertionError, match="`unet_lora_config` should be None"):
            _ = MODELS.build(cfg)

        cfg = self._get_config()
        cfg.update(text_encoder_lora_config=dict(type="dummy"))
        with pytest.raises(
                AssertionError, match="`text_encoder_lora_config` should be None"):
            _ = MODELS.build(cfg)

        cfg = self._get_config()
        cfg.update(finetune_text_encoder=True)
        with pytest.raises(
                AssertionError,
                match="`finetune_text_encoder` should be False"):
            _ = MODELS.build(cfg)

    def test_infer(self):
        cfg = self._get_config()
        StableDiffuser = MODELS.build(cfg)

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
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
            negative_prompt="noise",
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            output_type="latent",
            height=64,
            width=64)
        assert len(result) == 1
        assert type(result[0]) == torch.Tensor
        assert result[0].shape == (4, 32, 32)

        # test infer with hidden_states_idx=-1
        cfg = self._get_config()
        cfg.update(hidden_states_idx=-1)
        StableDiffuser = MODELS.build(cfg)
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            negative_prompt="noise",
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
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                text=["a dog"],
                clip_img=[torch.zeros((3, 32, 32))],
                time_ids=[torch.zeros((1, 6))]))
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
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                text=["a dog"],
                clip_img=[torch.zeros((3, 32, 32))],
                time_ids=[torch.zeros((1, 6))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_last_hidden(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(hidden_states_idx=-1)
        StableDiffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                text=["a dog"],
                clip_img=[torch.zeros((3, 32, 32))],
                time_ids=[torch.zeros((1, 6))]))
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
