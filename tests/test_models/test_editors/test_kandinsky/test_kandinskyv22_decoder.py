from unittest import TestCase

import pytest
import torch
from diffusers import DDPMScheduler, UNet2DConditionModel, VQModel
from mmengine.optim import OptimWrapper
from torch.optim import SGD
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

from diffengine.models.editors import (
    KandinskyV22Decoder,
    KandinskyV22DecoderDataPreprocessor,
)
from diffengine.models.losses import DeBiasEstimationLoss, L2Loss, SNRL2Loss
from diffengine.registry import MODELS


class TestKandinskyV22Decoder(TestCase):

    def _get_config(self) -> dict:
        vae_kwargs = {
            "block_out_channels": [32, 64],
            "down_block_types": ["DownEncoderBlock2D",
                                 "AttnDownEncoderBlock2D"],
            "in_channels": 3,
            "latent_channels": 4,
            "layers_per_block": 1,
            "norm_num_groups": 8,
            "norm_type": "spatial",
            "num_vq_embeddings": 12,
            "out_channels": 3,
            "up_block_types": [
                "AttnUpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
            "vq_embed_dim": 4,
        }
        model_kwargs = {
            "in_channels": 4,
            "out_channels": 8,
            "addition_embed_type": "image",
            "down_block_types": (
                "ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"),
            "up_block_types": (
                "SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"),
            "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
            "block_out_channels": (32, 32 * 2),
            "layers_per_block": 1,
            "encoder_hid_dim": 32,
            "encoder_hid_dim_type": "image_proj",
            "cross_attention_dim": 32,
            "attention_head_dim": 4,
            "resnet_time_scale_shift": "scale_shift",
            "class_embed_type": None,
        }
        config = CLIPVisionConfig(
            hidden_size=32,
            image_size=224,
            projection_dim=32,
            intermediate_size=37,
            num_attention_heads=4,
            num_channels=3,
            num_hidden_layers=5,
            patch_size=14,
        )
        return dict(
            type=KandinskyV22Decoder,
             scheduler=dict(type=DDPMScheduler),
             image_encoder=dict(
                 type=CLIPVisionModelWithProjection,
                 config=config),
             vae=dict(type=VQModel, **vae_kwargs),
             unet=dict(type=UNet2DConditionModel, **model_kwargs),
            data_preprocessor=dict(type=KandinskyV22DecoderDataPreprocessor),
            loss=dict(type=L2Loss))

    def test_train_step(self):
        # test load with loss module
        cfg = self._get_config()
        StableDiffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"],
                        clip_img=[torch.zeros((3, 224, 224))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_lora(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(unet_lora_config=dict(
                type="LoRA", r=4,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]))
        StableDiffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"],
                        clip_img=[torch.zeros((3, 224, 224))]))
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
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"],
                        clip_img=[torch.zeros((3, 224, 224))]))
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
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"],
                        clip_img=[torch.zeros((3, 224, 224))]))
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
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a sks dog"],
                        clip_img=[torch.zeros((3, 224, 224))]))
        data["inputs"]["result_class_image"] = dict(
            img=[torch.zeros((3, 64, 64))],
            text=["a dog"],
            clip_img=[torch.zeros((3, 224, 224))])  # type: ignore[assignment]
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
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"],
                        clip_img=[torch.zeros((3, 224, 224))]))
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
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"],
                        clip_img=[torch.zeros((3, 224, 224))]))
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
