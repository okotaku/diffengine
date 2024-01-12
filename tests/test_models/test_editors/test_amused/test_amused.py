from unittest import TestCase

import pytest
import torch
from diffusers import (
    UVit2DModel,
    VQModel,
)
from mmengine.optim import OptimWrapper
from torch.optim import SGD
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer

from diffengine.models.editors import AMUSEd, AMUSEdPreprocessor
from diffengine.models.losses import CrossEntropyLoss
from diffengine.registry import MODELS


class TestPixArtAlpha(TestCase):

    def _get_config(self) -> dict:
        base_model = "amused/amused-256"
        text_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=64,
            layer_norm_eps=1e-05,
            num_attention_heads=8,
            num_hidden_layers=3,
            pad_token_id=1,
            vocab_size=1000,
            projection_dim=32,
        )
        return dict(
            type=AMUSEd,
             model=base_model,
             tokenizer=dict(
                type=CLIPTokenizer.from_pretrained,
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip"),
             text_encoder=dict(type=CLIPTextModelWithProjection,
                               config=text_config),
             vae=dict(
                type=VQModel,
                act_fn="silu",
                block_out_channels=[32],
                down_block_types=[
                    "DownEncoderBlock2D",
                ],
                in_channels=3,
                latent_channels=32,
                layers_per_block=2,
                norm_num_groups=32,
                num_vq_embeddings=32,
                out_channels=3,
                sample_size=32,
                up_block_types=[
                    "UpDecoderBlock2D",
                ],
                mid_block_add_attention=False,
                lookup_from_codebook=True),
             transformer=dict(type=UVit2DModel,
                hidden_size=32,
                use_bias=False,
                hidden_dropout=0.0,
                cond_embed_dim=32,
                micro_cond_encode_dim=2,
                micro_cond_embed_dim=10,
                encoder_hidden_size=32,
                vocab_size=32,
                codebook_size=32,
                in_channels=32,
                block_out_channels=32,
                num_res_blocks=1,
                downsample=True,
                upsample=True,
                block_num_heads=1,
                num_hidden_layers=1,
                num_attention_heads=1,
                attention_dropout=0.0,
                intermediate_size=32,
                layer_norm_eps=1e-06,
                ln_elementwise_affine=True),
            data_preprocessor=dict(type=AMUSEdPreprocessor),
            loss=dict(type=CrossEntropyLoss))

    def test_init(self):
        cfg = self._get_config()
        cfg.update(text_encoder_lora_config=dict(type="dummy"))
        with pytest.raises(
                AssertionError, match="If you want to use LoRA"):
            _ = MODELS.build(cfg)

        cfg = self._get_config()
        cfg.update(transformer_lora_config=dict(type="dummy"),
                   finetune_text_encoder=True)
        with pytest.raises(
                AssertionError, match="If you want to finetune text"):
            _ = MODELS.build(cfg)

    def test_train_step(self):
        # test load with loss module
        cfg = self._get_config()
        StableDiffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"],
                micro_conds=[torch.zeros((1, 5))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_lora(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(
            transformer_lora_config=dict(
                type="LoRA", r=4,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config = dict(
                type="LoRA", r=4,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
        )
        StableDiffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"],
                micro_conds=[torch.zeros((1, 5))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_gradient_checkpointing(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(
            gradient_checkpointing=True,
        )
        StableDiffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"],
                micro_conds=[torch.zeros((1, 5))]))
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
                micro_conds=[torch.zeros((1, 5))]))
        data["inputs"]["result_class_image"] = dict(
            img=[torch.zeros((3, 64, 64))],
            text=["a dog"],
            micro_conds=[torch.zeros((1, 5))])  # type: ignore[assignment]
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
