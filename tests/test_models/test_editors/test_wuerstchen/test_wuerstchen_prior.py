from copy import deepcopy
from unittest import TestCase

import pytest
import torch
from diffusers import DDPMWuerstchenScheduler
from diffusers.pipelines.wuerstchen import WuerstchenPrior
from mmengine import print_log
from mmengine.optim import OptimWrapper
from torch import nn
from torch.optim import SGD
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffengine.models.editors import SDDataPreprocessor, WuerstchenPriorModel
from diffengine.models.editors.wuerstchen.efficient_net_encoder import (
    EfficientNetEncoder,
)
from diffengine.models.losses import DeBiasEstimationLoss, L2Loss, SNRL2Loss
from diffengine.registry import MODELS


class DummyWuerstchenPriorModel(WuerstchenPriorModel):
    def __init__(
        self,
        decoder_model: str = "warp-ai/wuerstchen",
        prior_model: str = "warp-ai/wuerstchen-prior",
        loss: dict | None = None,
        prior_lora_config: dict | None = None,
        text_encoder_lora_config: dict | None = None,
        prior_loss_weight: float = 1.,
        data_preprocessor: dict | nn.Module | None = None,
        noise_generator: dict | None = None,
        timesteps_generator: dict | None = None,
        input_perturbation_gamma: float = 0.0,
        *,
        finetune_text_encoder: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "SDDataPreprocessor"}
        if noise_generator is None:
            noise_generator = {"type": "WhiteNoise"}
        if timesteps_generator is None:
            timesteps_generator = {"type": "WuerstchenRandomTimeSteps"}
        if loss is None:
            loss = {"type": "L2Loss", "loss_weight": 1.0}
        super(WuerstchenPriorModel, self).__init__(data_preprocessor=data_preprocessor)
        if (
            prior_lora_config is not None) and (
                text_encoder_lora_config is not None) and (
                    not finetune_text_encoder):
                print_log(
                    "You are using LoRA for Prior and text encoder. "
                    "But you are not set `finetune_text_encoder=True`. "
                    "We will set `finetune_text_encoder=True` for you.")
                finetune_text_encoder = True
        if text_encoder_lora_config is not None:
            assert finetune_text_encoder, (
                "If you want to use LoRA for text encoder, "
                "you should set finetune_text_encoder=True."
            )
        if finetune_text_encoder and prior_lora_config is not None:
            assert text_encoder_lora_config is not None, (
                "If you want to finetune text encoder with LoRA Prior, "
                "you should set text_encoder_lora_config."
            )

        self.decoder_model = decoder_model
        self.prior_model = prior_model
        self.prior_lora_config = deepcopy(prior_lora_config)
        self.text_encoder_lora_config = deepcopy(text_encoder_lora_config)
        self.finetune_text_encoder = finetune_text_encoder
        self.prior_loss_weight = prior_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.input_perturbation_gamma = input_perturbation_gamma

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module: nn.Module = loss
        assert not self.loss_module.use_snr, \
            "WuerstchenPriorModel does not support SNR loss."

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")
        config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        self.text_encoder = CLIPTextModel(config)

        self.image_encoder = EfficientNetEncoder(c_latent=2)

        self.scheduler = DDPMWuerstchenScheduler()

        model_kwargs = {
            "c_in": 2,
            "c": 8,
            "depth": 2,
            "c_cond": 32,
            "c_r": 8,
            "nhead": 2,
        }
        self.prior = WuerstchenPrior(**model_kwargs)
        self.noise_generator = MODELS.build(noise_generator)
        self.timesteps_generator = MODELS.build(timesteps_generator)
        self.prepare_model()
        self.set_lora()


class TestWuerstchenPrior(TestCase):

    def test_init(self):
        with pytest.raises(
                AssertionError, match="If you want to use LoRA"):
            _ = DummyWuerstchenPriorModel(
                text_encoder_lora_config=dict(type="dummy"),
                data_preprocessor=SDDataPreprocessor())

        with pytest.raises(
                AssertionError, match="If you want to finetune text"):
            _ = DummyWuerstchenPriorModel(
                prior_lora_config=dict(type="dummy"),
                finetune_text_encoder=True,
                data_preprocessor=SDDataPreprocessor())

    def test_train_step(self):
        # test load with loss module
        StableDiffuser = DummyWuerstchenPriorModel(
            loss=L2Loss(),
            data_preprocessor=SDDataPreprocessor())

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
        StableDiffuser = DummyWuerstchenPriorModel(
            loss=L2Loss(),
            prior_lora_config=dict(
                type="LoRA", r=4,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config = dict(
                type="LoRA", r=4,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
            data_preprocessor=SDDataPreprocessor())

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
        StableDiffuser = DummyWuerstchenPriorModel(
            input_perturbation_gamma=0.1,
            loss=L2Loss(),
            data_preprocessor=SDDataPreprocessor())

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
        StableDiffuser = DummyWuerstchenPriorModel(
            loss=L2Loss(),
            data_preprocessor=SDDataPreprocessor(),
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
        StableDiffuser = DummyWuerstchenPriorModel(
            loss=L2Loss(),
            data_preprocessor=SDDataPreprocessor())

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
        with pytest.raises(
                AssertionError, match="WuerstchenPriorModel does not support"):
            _ = DummyWuerstchenPriorModel(
                loss=SNRL2Loss(),
                data_preprocessor=SDDataPreprocessor())

    def test_train_step_debias_estimation_loss(self):
        # test load with loss module
        with pytest.raises(
                AssertionError, match="WuerstchenPriorModel does not support"):
            _ = DummyWuerstchenPriorModel(
                loss=DeBiasEstimationLoss(),
                data_preprocessor=SDDataPreprocessor())

    def test_val_and_test_step(self):
        StableDiffuser = DummyWuerstchenPriorModel(
            loss=L2Loss(),
            data_preprocessor=SDDataPreprocessor())

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            StableDiffuser.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            StableDiffuser.test_step(torch.zeros((1, )))
