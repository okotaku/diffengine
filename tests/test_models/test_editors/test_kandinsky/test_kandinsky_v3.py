from copy import deepcopy
from unittest import TestCase

import pytest
import torch
from diffusers import DDPMScheduler, Kandinsky3UNet, VQModel
from mmengine.optim import OptimWrapper
from torch import nn
from torch.optim import SGD
from transformers import AutoTokenizer, T5EncoderModel

from diffengine.models.editors import (
    KandinskyV3,
    SDDataPreprocessor,
)
from diffengine.models.losses import DeBiasEstimationLoss, L2Loss, SNRL2Loss
from diffengine.registry import MODELS


class DummyKandinskyV3(KandinskyV3):
    def __init__(
        self,
        model: str = "kandinsky-community/kandinsky-3",
        loss: dict | None = None,
        unet_lora_config: dict | None = None,
        prior_loss_weight: float = 1.,
        tokenizer_max_length: int = 128,
        prediction_type: str | None = None,
        data_preprocessor: dict | nn.Module | None = None,
        noise_generator: dict | None = None,
        timesteps_generator: dict | None = None,
        input_perturbation_gamma: float = 0.0,
        *,
        gradient_checkpointing: bool = False,
    ) -> None:
        assert gradient_checkpointing is False, (
            "KandinskyV3 does not support gradient checkpointing.")
        if data_preprocessor is None:
            data_preprocessor = {"type": "SDDataPreprocessor"}
        if noise_generator is None:
            noise_generator = {"type": "WhiteNoise"}
        if timesteps_generator is None:
            timesteps_generator = {"type": "TimeSteps"}
        if loss is None:
            loss = {"type": "L2Loss", "loss_weight": 1.0}
        super(KandinskyV3, self).__init__(data_preprocessor=data_preprocessor)

        self.model = model
        self.unet_lora_config = deepcopy(unet_lora_config)
        self.prior_loss_weight = prior_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.tokenizer_max_length = tokenizer_max_length
        self.input_perturbation_gamma = input_perturbation_gamma

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module: nn.Module = loss

        self.prediction_type = prediction_type

        self.tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-t5")

        self.text_encoder = T5EncoderModel.from_pretrained(
            "hf-internal-testing/tiny-random-t5")

        self.scheduler = DDPMScheduler()

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
        self.vae = VQModel(**vae_kwargs)

        self.unet = Kandinsky3UNet(
            in_channels=4,
            time_embedding_dim=4,
            groups=2,
            attention_head_dim=4,
            layers_per_block=3,
            block_out_channels=(32, 64),
            cross_attention_dim=4,
            encoder_hid_dim=32,
        )
        self.noise_generator = MODELS.build(noise_generator)
        self.timesteps_generator = MODELS.build(timesteps_generator)

        self.prepare_model()
        self.set_lora()


class TestKandinskyV3(TestCase):

    def test_init(self):
        with pytest.raises(
                AssertionError, match="KandinskyV3 does not support gradient"):
            _ = DummyKandinskyV3(
                    loss=L2Loss(),
                    data_preprocessor=SDDataPreprocessor(),
                    gradient_checkpointing=True)

    def test_train_step(self):
        # test load with loss module
        StableDiffuser = DummyKandinskyV3(
            loss=L2Loss(),
            data_preprocessor=SDDataPreprocessor())

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
        StableDiffuser = DummyKandinskyV3(
            loss=L2Loss(),
            unet_lora_config=dict(
                type="LoRA", r=4,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            data_preprocessor=SDDataPreprocessor())

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
        StableDiffuser = DummyKandinskyV3(
            input_perturbation_gamma=0.1,
            loss=L2Loss(),
            data_preprocessor=SDDataPreprocessor())

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
        StableDiffuser = DummyKandinskyV3(
            loss=L2Loss(),
            data_preprocessor=SDDataPreprocessor())

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
        StableDiffuser = DummyKandinskyV3(
            loss=SNRL2Loss(),
            data_preprocessor=SDDataPreprocessor())

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
        StableDiffuser = DummyKandinskyV3(
            loss=DeBiasEstimationLoss(),
            data_preprocessor=SDDataPreprocessor())

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
        StableDiffuser = DummyKandinskyV3(
            loss=L2Loss(),
            data_preprocessor=SDDataPreprocessor())

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            StableDiffuser.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            StableDiffuser.test_step(torch.zeros((1, )))
