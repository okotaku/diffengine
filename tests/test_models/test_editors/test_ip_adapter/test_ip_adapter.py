from unittest import TestCase

import pytest
import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from mmengine.optim import OptimWrapper
from torch.optim import SGD
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from diffengine.models.editors import IPAdapterXL as Base
from diffengine.models.editors import IPAdapterXLDataPreprocessor
from diffengine.models.editors.ip_adapter.image_projection import ImageProjModel
from diffengine.models.losses import L2Loss
from diffengine.registry import MODELS


class IPAdapterXL(Base):
    def _encode_image(self, image, num_images_per_prompt):
        if not isinstance(image, torch.Tensor):
            from transformers import CLIPImageProcessor
            image_processor = CLIPImageProcessor.from_pretrained(
                "hf-internal-testing/unidiffuser-diffusers-test",
                subfolder="image_processor")
            image = image_processor(image, return_tensors="pt").pixel_values

        image = image.to(device=self.device)
        image_embeddings = self.image_encoder(image).image_embeds
        image_prompt_embeds = self.image_projection(image_embeddings)
        uncond_image_prompt_embeds = self.image_projection(
            torch.zeros_like(image_embeddings))

        # duplicate image embeddings for each generation per prompt, using mps
        # friendly method
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(
            1, num_images_per_prompt, 1)
        image_prompt_embeds = image_prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
            1, num_images_per_prompt, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1)

        return image_prompt_embeds, uncond_image_prompt_embeds


class TestIPAdapterXL(TestCase):

    def _get_config(self) -> dict:
        base_model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        return dict(type=IPAdapterXL,
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
             image_projection=dict(type=ImageProjModel,
                                   clip_extra_context_tokens=4),
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

    def test_val_and_test_step(self):
        cfg = self._get_config()
        StableDiffuser = MODELS.build(cfg)

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            StableDiffuser.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            StableDiffuser.test_step(torch.zeros((1, )))
