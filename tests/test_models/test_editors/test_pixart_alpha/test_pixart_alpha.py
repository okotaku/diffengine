from copy import deepcopy
from unittest import TestCase

import pytest
import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    Transformer2DModel,
)
from mmengine import print_log
from mmengine.optim import OptimWrapper
from torch import nn
from torch.optim import SGD
from transformers import AutoTokenizer, T5EncoderModel

from diffengine.models.editors import PixArtAlpha, PixArtAlphaDataPreprocessor
from diffengine.models.losses import DeBiasEstimationLoss, L2Loss, SNRL2Loss
from diffengine.registry import MODELS


class DummyPixArtAlpha(PixArtAlpha):
    def __init__(
        self,
        model: str = "PixArt-alpha/PixArt-XL-2-1024-MS",
        loss: dict | None = None,
        transformer_lora_config: dict | None = None,
        text_encoder_lora_config: dict | None = None,
        prior_loss_weight: float = 1.,
        tokenizer_max_length: int = 120,
        prediction_type: str | None = None,
        data_preprocessor: dict | nn.Module | None = None,
        noise_generator: dict | None = None,
        timesteps_generator: dict | None = None,
        input_perturbation_gamma: float = 0.0,
        *,
        finetune_text_encoder: bool = False,
        gradient_checkpointing: bool = False,
        enable_xformers: bool = False,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "PixArtAlphaDataPreprocessor"}
        if noise_generator is None:
            noise_generator = {"type": "WhiteNoise"}
        if timesteps_generator is None:
            timesteps_generator = {"type": "TimeSteps"}
        if loss is None:
            loss = {"type": "L2Loss", "loss_weight": 1.0}
        super(PixArtAlpha, self).__init__(data_preprocessor=data_preprocessor)
        if (
            transformer_lora_config is not None) and (
                text_encoder_lora_config is not None) and (
                    not finetune_text_encoder):
                print_log(
                    "You are using LoRA for Transformer and text encoder. "
                    "But you are not set `finetune_text_encoder=True`. "
                    "We will set `finetune_text_encoder=True` for you.")
                finetune_text_encoder = True
        if text_encoder_lora_config is not None:
            assert finetune_text_encoder, (
                "If you want to use LoRA for text encoder, "
                "you should set finetune_text_encoder=True."
            )
        if finetune_text_encoder and transformer_lora_config is not None:
            assert text_encoder_lora_config is not None, (
                "If you want to finetune text encoder with LoRA Transformer, "
                "you should set text_encoder_lora_config."
            )

        self.model = model
        self.transformer_lora_config = deepcopy(transformer_lora_config)
        self.text_encoder_lora_config = deepcopy(text_encoder_lora_config)
        self.finetune_text_encoder = finetune_text_encoder
        self.prior_loss_weight = prior_loss_weight
        self.tokenizer_max_length = tokenizer_max_length
        self.gradient_checkpointing = gradient_checkpointing
        self.input_perturbation_gamma = input_perturbation_gamma
        self.enable_xformers = enable_xformers

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module: nn.Module = loss

        assert prediction_type in [None, "epsilon", "v_prediction"]
        self.prediction_type = prediction_type

        self.tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-t5")
        self.scheduler = DDPMScheduler.from_pretrained(
            model, subfolder="scheduler")

        self.transformer = Transformer2DModel(
            sample_size=8,
            num_layers=2,
            patch_size=2,
            attention_head_dim=8,
            num_attention_heads=3,
            caption_channels=32,
            in_channels=4,
            cross_attention_dim=24,
            out_channels=8,
            attention_bias=True,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            norm_type="ada_norm_single",
            norm_elementwise_affine=False,
            norm_eps=1e-6,
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            "hf-internal-testing/tiny-random-t5")
        self.vae = AutoencoderKL()
        self.noise_generator = MODELS.build(noise_generator)
        self.timesteps_generator = MODELS.build(timesteps_generator)
        self.prepare_model()
        self.set_lora()
        self.set_xformers()


class TestPixArtAlpha(TestCase):

    def test_init(self):
        with pytest.raises(
                AssertionError, match="If you want to use LoRA"):
            _ = DummyPixArtAlpha(
                text_encoder_lora_config=dict(type="dummy"),
                data_preprocessor=PixArtAlphaDataPreprocessor())

        with pytest.raises(
                AssertionError, match="If you want to finetune text"):
            _ = DummyPixArtAlpha(
                transformer_lora_config=dict(type="dummy"),
                finetune_text_encoder=True,
                data_preprocessor=PixArtAlphaDataPreprocessor())

    def test_train_step(self):
        # test load with loss module
        StableDiffuser = DummyPixArtAlpha(
            loss=L2Loss(),
            data_preprocessor=PixArtAlphaDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_lora(self):
        # test load with loss module
        StableDiffuser = DummyPixArtAlpha(
            loss=L2Loss(),
            transformer_lora_config=dict(
                type="LoRA", r=4,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config = dict(
                type="LoRA", r=4,
                target_modules=["q", "k", "v", "o"]),
            data_preprocessor=PixArtAlphaDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_input_perturbation(self):
        # test load with loss module
        StableDiffuser = DummyPixArtAlpha(
            input_perturbation_gamma=0.1,
            loss=L2Loss(),
            data_preprocessor=PixArtAlphaDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_gradient_checkpointing(self):
        # test load with loss module
        StableDiffuser = DummyPixArtAlpha(
            loss=L2Loss(),
            data_preprocessor=PixArtAlphaDataPreprocessor(),
            gradient_checkpointing=True)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_dreambooth(self):
        # test load with loss module
        StableDiffuser = DummyPixArtAlpha(
            loss=L2Loss(),
            data_preprocessor=PixArtAlphaDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a sks dog"]))
        data["inputs"]["result_class_image"] = dict(
            img=[torch.zeros((3, 64, 64))],
            text=["a dog"])  # type: ignore[assignment]
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_snr_loss(self):
        # test load with loss module
        StableDiffuser = DummyPixArtAlpha(
            loss=SNRL2Loss(),
            data_preprocessor=PixArtAlphaDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_debias_estimation_loss(self):
        # test load with loss module
        StableDiffuser = DummyPixArtAlpha(
            loss=DeBiasEstimationLoss(),
            data_preprocessor=PixArtAlphaDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_val_and_test_step(self):
        StableDiffuser = DummyPixArtAlpha(
            loss=L2Loss(),
            data_preprocessor=PixArtAlphaDataPreprocessor())

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            StableDiffuser.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            StableDiffuser.test_step(torch.zeros((1, )))
