import os
from copy import deepcopy
from unittest import TestCase

import pytest
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetXSModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from mmengine import print_log
from mmengine.optim import OptimWrapper
from torch import nn
from torch.optim import SGD
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from diffengine.models.editors import (
    SDXLControlNetDataPreprocessor,
    StableDiffusionXL,
    StableDiffusionXLControlNetXS,
)
from diffengine.models.losses import L2Loss
from diffengine.registry import MODELS

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class DummyStableDiffusionXLControlNetXS(StableDiffusionXLControlNetXS):
    def __init__(  # noqa: C901,PLR0915
        self,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet_model: str | None = None,
        transformer_layers_per_block: list[int] | None = None,
        loss: dict | None = None,
        unet_lora_config: dict | None = None,
        text_encoder_lora_config: dict | None = None,
        prior_loss_weight: float = 1.,
        prediction_type: str | None = None,
        data_preprocessor: dict | nn.Module | None = None,
        noise_generator: dict | None = None,
        timesteps_generator: dict | None = None,
        input_perturbation_gamma: float = 0.0,
        *,
        finetune_text_encoder: bool = False,
        gradient_checkpointing: bool = False,
        pre_compute_text_embeddings: bool = False,
        enable_xformers: bool = False,
    ) -> None:
        assert unet_lora_config is None, \
            "`unet_lora_config` should be None when training ControlNet"
        assert text_encoder_lora_config is None, \
            "`text_encoder_lora_config` should be None when training ControlNet"
        assert not finetune_text_encoder, \
            "`finetune_text_encoder` should be False when training ControlNet"
        self.controlnet_model = controlnet_model
        self.transformer_layers_per_block = transformer_layers_per_block

        if data_preprocessor is None:
            data_preprocessor = {"type": "SDXLDataPreprocessor"}
        if noise_generator is None:
            noise_generator = {"type": "WhiteNoise"}
        if timesteps_generator is None:
            timesteps_generator = {"type": "TimeSteps"}
        if loss is None:
            loss = {"type": "L2Loss", "loss_weight": 1.0}
        super(StableDiffusionXL, self).__init__(
            data_preprocessor=data_preprocessor)

        if (
            unet_lora_config is not None) and (
                text_encoder_lora_config is not None) and (
                    not finetune_text_encoder):
                print_log(
                    "You are using LoRA for Unet and text encoder. "
                    "But you are not set `finetune_text_encoder=True`. "
                    "We will set `finetune_text_encoder=True` for you.")
                finetune_text_encoder = True
        if text_encoder_lora_config is not None:
            assert finetune_text_encoder, (
                "If you want to use LoRA for text encoder, "
                "you should set finetune_text_encoder=True."
            )
        if finetune_text_encoder and unet_lora_config is not None:
            assert text_encoder_lora_config is not None, (
                "If you want to finetune text encoder with LoRA Unet, "
                "you should set text_encoder_lora_config."
            )
        if pre_compute_text_embeddings:
            assert not finetune_text_encoder

        self.model = model
        self.unet_lora_config = deepcopy(unet_lora_config)
        self.text_encoder_lora_config = deepcopy(text_encoder_lora_config)
        self.finetune_text_encoder = finetune_text_encoder
        self.prior_loss_weight = prior_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.pre_compute_text_embeddings = pre_compute_text_embeddings
        self.input_perturbation_gamma = input_perturbation_gamma
        self.enable_xformers = enable_xformers

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module: nn.Module = loss

        assert prediction_type in [None, "epsilon", "v_prediction"]
        self.prediction_type = prediction_type

        if not self.pre_compute_text_embeddings:
            text_encoder_config = CLIPTextConfig(
                bos_token_id=0,
                eos_token_id=2,
                hidden_size=32,
                intermediate_size=37,
                layer_norm_eps=1e-05,
                num_attention_heads=4,
                num_hidden_layers=5,
                pad_token_id=1,
                vocab_size=1000,
                # SD2-specific config below
                hidden_act="gelu",
                projection_dim=32,
            )
            self.tokenizer_one = CLIPTokenizer.from_pretrained(
                "hf-internal-testing/tiny-random-clip")
            self.tokenizer_two = CLIPTokenizer.from_pretrained(
                "hf-internal-testing/tiny-random-clip")

            self.text_encoder_one = CLIPTextModel(text_encoder_config)
            self.text_encoder_two = CLIPTextModelWithProjection(text_encoder_config)

        self.scheduler = DDPMScheduler.from_pretrained(
            model, subfolder="scheduler")

        self.vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        self.unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        self.noise_generator = MODELS.build(noise_generator)
        self.timesteps_generator = MODELS.build(timesteps_generator)
        self.prepare_model()
        self.set_lora()
        self.set_xformers()

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.controlnet = ControlNetXSModel.from_unet(
            self.unet,
            time_embedding_mix=0.95,
            learn_embedding=True,
            size_ratio=0.5,
            conditioning_embedding_out_channels=(16, 32),
        )

        if self.gradient_checkpointing:
            # TODO(takuoko): Support ControlNetXSModel for gradient  # noqa
            # checkpointing
            # self.controlnet.enable_gradient_checkpointing()
            self.unet.enable_gradient_checkpointing()

        self.vae.requires_grad_(requires_grad=False)
        print_log("Set VAE untrainable.", "current")
        self.text_encoder_one.requires_grad_(requires_grad=False)
        self.text_encoder_two.requires_grad_(requires_grad=False)
        print_log("Set Text Encoder untrainable.", "current")
        self.unet.requires_grad_(requires_grad=False)
        print_log("Set Unet untrainable.", "current")


@pytest.mark.skipif(IN_GITHUB_ACTIONS,
                    reason=("Test doesn't work because ControlNetXS hasn't"
                            " supported peft>=0.7 yet."))
class TestStableDiffusionXLControlNet(TestCase):

    def test_init(self):
        with pytest.raises(
                AssertionError, match="`unet_lora_config` should be None"):
            _ = DummyStableDiffusionXLControlNetXS(
                "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
                data_preprocessor=SDXLControlNetDataPreprocessor(),
                unet_lora_config=dict(type="dummy"))

        with pytest.raises(
                AssertionError, match="`text_encoder_lora_config` should be None"):
            _ = DummyStableDiffusionXLControlNetXS(
                "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
                data_preprocessor=SDXLControlNetDataPreprocessor(),
                text_encoder_lora_config=dict(type="dummy"))

        with pytest.raises(
                AssertionError,
                match="`finetune_text_encoder` should be False"):
            _ = DummyStableDiffusionXLControlNetXS(
                "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
                data_preprocessor=SDXLControlNetDataPreprocessor(),
                finetune_text_encoder=True)

    def test_infer(self):
        StableDiffuser = DummyStableDiffusionXLControlNetXS(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            data_preprocessor=SDXLControlNetDataPreprocessor())

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

        # output_type = 'latent'
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
        StableDiffuser = DummyStableDiffusionXLControlNetXS(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            loss=L2Loss(),
            data_preprocessor=SDXLControlNetDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                text=["a dog"],
                time_ids=[torch.zeros((1, 6))],
                condition_img=[torch.zeros((3, 64, 64))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_gradient_checkpointing(self):
        # test load with loss module
        StableDiffuser = DummyStableDiffusionXLControlNetXS(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            loss=L2Loss(),
            data_preprocessor=SDXLControlNetDataPreprocessor(),
            gradient_checkpointing=True)

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                text=["a dog"],
                time_ids=[torch.zeros((1, 6))],
                condition_img=[torch.zeros((3, 64, 64))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_val_and_test_step(self):
        StableDiffuser = DummyStableDiffusionXLControlNetXS(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            loss=L2Loss(),
            data_preprocessor=SDXLControlNetDataPreprocessor())

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            StableDiffuser.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            StableDiffuser.test_step(torch.zeros((1, )))
