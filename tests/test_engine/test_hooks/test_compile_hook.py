import copy

from diffusers import AutoencoderKL, Transformer2DModel, UNet2DConditionModel
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from mmengine.testing.runner_test_case import ToyModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, T5EncoderModel

from diffengine.engine.hooks import CompileHook
from diffengine.models.editors import (
    SDControlNetDataPreprocessor,
    SDDataPreprocessor,
    SDXLControlNetDataPreprocessor,
    SDXLDataPreprocessor,
    StableDiffusion,
    StableDiffusionControlNet,
    StableDiffusionXL,
    StableDiffusionXLControlNet,
    StableDiffusionXLT2IAdapter,
)
from diffengine.models.losses import L2Loss
from diffengine.models.utils import CubicSamplingTimeSteps, TimeSteps, WhiteNoise


class ToyModelPixArt(ToyModel):

    def __init__(self) -> None:
        super().__init__()
        height = 12
        width = 12

        model_kwargs = {
            "attention_bias": True,
            "cross_attention_dim": 32,
            "attention_head_dim": height * width,
            "num_attention_heads": 1,
            "num_vector_embeds": 12,
            "num_embeds_ada_norm": 12,
            "norm_num_groups": 32,
            "sample_size": width,
            "activation_fn": "geglu-approximate",
        }

        self.transformer = Transformer2DModel(**model_kwargs)
        self.text_encoder = T5EncoderModel.from_pretrained(
            "hf-internal-testing/tiny-random-t5")
        self.vae = AutoencoderKL(
            in_channels=4,
            out_channels=4,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(32,),
            layers_per_block=1,
            act_fn="silu",
            latent_channels=4,
            norm_num_groups=16,
            sample_size=16,
        )
        self.finetune_text_encoder = True

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class TestCompileHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="StableDiffusion", module=StableDiffusion)
        MODELS.register_module(
            name="StableDiffusionXL", module=StableDiffusionXL)
        MODELS.register_module(
            name="SDDataPreprocessor", module=SDDataPreprocessor)
        MODELS.register_module(
            name="SDXLDataPreprocessor", module=SDXLDataPreprocessor)
        MODELS.register_module(
            name="StableDiffusionControlNet", module=StableDiffusionControlNet)
        MODELS.register_module(name="SDControlNetDataPreprocessor",
                               module=SDControlNetDataPreprocessor)
        MODELS.register_module(name="StableDiffusionXLControlNet",
                               module=StableDiffusionXLControlNet)
        MODELS.register_module(name="SDXLControlNetDataPreprocessor",
                               module=SDXLControlNetDataPreprocessor)
        MODELS.register_module(name="StableDiffusionXLT2IAdapter",
                               module=StableDiffusionXLT2IAdapter)
        MODELS.register_module(name="L2Loss", module=L2Loss)
        MODELS.register_module(name="WhiteNoise", module=WhiteNoise)
        MODELS.register_module(name="TimeSteps", module=TimeSteps)
        MODELS.register_module(
            name="CubicSamplingTimeSteps", module=CubicSamplingTimeSteps)
        MODELS.register_module(name="ToyModelPixArt", module=ToyModelPixArt)
        return super().setUp()

    def tearDown(self) -> None:
        MODELS.module_dict.pop("StableDiffusion")
        MODELS.module_dict.pop("StableDiffusionXL")
        MODELS.module_dict.pop("SDDataPreprocessor")
        MODELS.module_dict.pop("SDXLDataPreprocessor")
        MODELS.module_dict.pop("StableDiffusionControlNet")
        MODELS.module_dict.pop("SDControlNetDataPreprocessor")
        MODELS.module_dict.pop("StableDiffusionXLControlNet")
        MODELS.module_dict.pop("SDXLControlNetDataPreprocessor")
        MODELS.module_dict.pop("StableDiffusionXLT2IAdapter")
        MODELS.module_dict.pop("L2Loss")
        MODELS.module_dict.pop("WhiteNoise")
        MODELS.module_dict.pop("TimeSteps")
        MODELS.module_dict.pop("CubicSamplingTimeSteps")
        MODELS.module_dict.pop("ToyModelPixArt")
        return super().tearDown()

    def test_init(self) -> None:
        CompileHook()

    def test_before_train(self) -> None:
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusion"
        cfg.model.model = "diffusers/tiny-stable-diffusion-torch"
        runner = self.build_runner(cfg)
        hook = CompileHook(compile_main=True)
        assert isinstance(runner.model.unet, UNet2DConditionModel)
        assert isinstance(runner.model.vae, AutoencoderKL)
        assert isinstance(runner.model.text_encoder, CLIPTextModel)
        # compile
        hook.before_train(runner)
        assert not isinstance(runner.model.unet, UNet2DConditionModel)
        assert not isinstance(runner.model.vae, AutoencoderKL)
        assert not isinstance(runner.model.text_encoder, CLIPTextModel)

        # Test StableDiffusionControlNet
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusionControlNet"
        cfg.model.model = "hf-internal-testing/tiny-stable-diffusion-pipe"
        cfg.model.controlnet_model="hf-internal-testing/tiny-controlnet"
        runner = self.build_runner(cfg)
        hook = CompileHook(compile_main=True)
        func = runner.model._forward_compile
        assert runner.model._forward_compile == func
        assert isinstance(runner.model.vae, AutoencoderKL)
        assert isinstance(runner.model.text_encoder, CLIPTextModel)
        # compile
        hook.before_train(runner)
        assert runner.model._forward_compile != func
        assert not isinstance(runner.model.vae, AutoencoderKL)
        assert not isinstance(runner.model.text_encoder, CLIPTextModel)

    def test_before_train_sdxl(self) -> None:
        # Test StableDiffusionXL
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusionXL"
        cfg.model.model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        runner = self.build_runner(cfg)
        hook = CompileHook(compile_main=True)
        assert isinstance(runner.model.unet, UNet2DConditionModel)
        assert isinstance(runner.model.vae, AutoencoderKL)
        assert isinstance(runner.model.text_encoder_one, CLIPTextModel)
        assert isinstance(
            runner.model.text_encoder_two, CLIPTextModelWithProjection)
        # compile
        hook.before_train(runner)
        assert not isinstance(runner.model.unet, UNet2DConditionModel)
        assert not isinstance(runner.model.vae, AutoencoderKL)
        assert not isinstance(runner.model.text_encoder_one, CLIPTextModel)
        assert not isinstance(
            runner.model.text_encoder_two, CLIPTextModelWithProjection)

        # Test StableDiffusionXLControlNet
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusionXLControlNet"
        cfg.model.model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        cfg.model.controlnet_model="hf-internal-testing/tiny-controlnet-sdxl"
        runner = self.build_runner(cfg)
        hook = CompileHook(compile_main=True)
        func = runner.model._forward_compile
        assert runner.model._forward_compile == func
        assert isinstance(runner.model.vae, AutoencoderKL)
        assert isinstance(runner.model.text_encoder_one, CLIPTextModel)
        assert isinstance(
            runner.model.text_encoder_two, CLIPTextModelWithProjection)
        # compile
        hook.before_train(runner)
        assert runner.model._forward_compile != func
        assert not isinstance(runner.model.vae, AutoencoderKL)
        assert not isinstance(runner.model.text_encoder_one, CLIPTextModel)
        assert not isinstance(
            runner.model.text_encoder_two, CLIPTextModelWithProjection)

        # Test StableDiffusionXLT2IAdapter
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusionXLT2IAdapter"
        cfg.model.model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        runner = self.build_runner(cfg)
        hook = CompileHook(compile_main=True)
        func = runner.model._forward_compile
        assert runner.model._forward_compile == func
        assert isinstance(runner.model.vae, AutoencoderKL)
        assert isinstance(runner.model.text_encoder_one, CLIPTextModel)
        assert isinstance(
            runner.model.text_encoder_two, CLIPTextModelWithProjection)
        # compile
        hook.before_train(runner)
        assert runner.model._forward_compile != func
        assert not isinstance(runner.model.vae, AutoencoderKL)
        assert not isinstance(runner.model.text_encoder_one, CLIPTextModel)
        assert not isinstance(
            runner.model.text_encoder_two, CLIPTextModelWithProjection)

    def test_before_train_pixart(self) -> None:
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "ToyModelPixArt"
        runner = self.build_runner(cfg)
        hook = CompileHook(compile_main=True)
        assert isinstance(runner.model.transformer, Transformer2DModel)
        assert isinstance(runner.model.vae, AutoencoderKL)
        assert isinstance(runner.model.text_encoder, T5EncoderModel)
        # compile
        hook.before_train(runner)
        assert not isinstance(runner.model.transformer, Transformer2DModel)
        assert not isinstance(runner.model.vae, AutoencoderKL)
        assert not isinstance(runner.model.text_encoder, T5EncoderModel)
