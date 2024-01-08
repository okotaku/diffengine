from unittest import TestCase

import pytest
import torch
from diffusers import DDPMWuerstchenScheduler
from diffusers.pipelines.wuerstchen import WuerstchenPrior
from mmengine.optim import OptimWrapper
from torch.optim import SGD
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffengine.models.editors import SDDataPreprocessor, WuerstchenPriorModel
from diffengine.models.editors.wuerstchen.efficient_net_encoder import (
    EfficientNetEncoder,
)
from diffengine.models.losses import DeBiasEstimationLoss, L2Loss, SNRL2Loss
from diffengine.registry import MODELS


class TestWuerstchenPrior(TestCase):

    def _get_config(self) -> dict:
        model_kwargs = {
            "c_in": 2,
            "c": 8,
            "depth": 2,
            "c_cond": 32,
            "c_r": 8,
            "nhead": 2,
        }
        text_config = CLIPTextConfig(
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
        return dict(
            type=WuerstchenPriorModel,
            tokenizer=dict(
                type=CLIPTokenizer.from_pretrained,
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip"),
             scheduler=dict(type=DDPMWuerstchenScheduler),
             text_encoder=dict(type=CLIPTextModel,
                               config=text_config),
             image_encoder=dict(type=EfficientNetEncoder, c_latent=2),
             prior=dict(type=WuerstchenPrior,
                        **model_kwargs),
            data_preprocessor=dict(type=SDDataPreprocessor),
            loss=dict(type=L2Loss))

    def test_init(self):
        cfg = self._get_config()
        cfg.update(text_encoder_lora_config=dict(type="dummy"))
        with pytest.raises(
                AssertionError, match="If you want to use LoRA"):
            _ = MODELS.build(cfg)

        cfg = self._get_config()
        cfg.update(prior_lora_config=dict(type="dummy"),
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
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_lora(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(prior_lora_config=dict(
                type="LoRA", r=4,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config = dict(
                type="LoRA", r=4,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
        StableDiffuser = MODELS.build(cfg)

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
        cfg = self._get_config()
        cfg.update(input_perturbation_gamma=0.1)
        StableDiffuser = MODELS.build(cfg)

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
        cfg = self._get_config()
        cfg.update(gradient_checkpointing=True)
        StableDiffuser = MODELS.build(cfg)

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
        cfg = self._get_config()
        StableDiffuser = MODELS.build(cfg)

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
        cfg = self._get_config()
        cfg.update(loss=dict(type=SNRL2Loss))
        with pytest.raises(
                AssertionError, match="WuerstchenPriorModel does not support"):
            _ = MODELS.build(cfg)

    def test_train_step_debias_estimation_loss(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(loss=dict(type=DeBiasEstimationLoss))
        with pytest.raises(
                AssertionError, match="WuerstchenPriorModel does not support"):
            _ = MODELS.build(cfg)

    def test_val_and_test_step(self):
        cfg = self._get_config()
        StableDiffuser = MODELS.build(cfg)

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            StableDiffuser.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            StableDiffuser.test_step(torch.zeros((1, )))
