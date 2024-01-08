import copy

from diffusers import AutoencoderKL, Transformer2DModel, UNet2DConditionModel
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from transformers import CLIPTextModel, CLIPTextModelWithProjection, T5EncoderModel

from diffengine.engine.hooks import CompileHook
from diffengine.models.utils import CubicSamplingTimeSteps, TimeSteps, WhiteNoise


class TestCompileHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="WhiteNoise", module=WhiteNoise)
        MODELS.register_module(name="TimeSteps", module=TimeSteps)
        MODELS.register_module(
            name="CubicSamplingTimeSteps", module=CubicSamplingTimeSteps)
        return super().setUp()

    def tearDown(self) -> None:
        MODELS.module_dict.pop("WhiteNoise")
        MODELS.module_dict.pop("TimeSteps")
        MODELS.module_dict.pop("CubicSamplingTimeSteps")
        return super().tearDown()

    def test_init(self) -> None:
        CompileHook()

    def test_before_train(self) -> None:
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/sd.py").model
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
        cfg.model = Config.fromfile("tests/configs/sdcn.py").model
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
        cfg.model = Config.fromfile("tests/configs/sdxl.py").model
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
        cfg.model = Config.fromfile("tests/configs/sdxlcn.py").model
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
        cfg.model = Config.fromfile("tests/configs/sdxl_t2iadapter.py").model
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
        cfg.model = Config.fromfile("tests/configs/pixart.py").model
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
