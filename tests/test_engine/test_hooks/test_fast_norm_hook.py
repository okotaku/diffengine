import copy
import unittest

from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from torch import nn

from diffengine.engine.hooks import FastNormHook
from diffengine.models.editors import (
    SDDataPreprocessor,
    SDXLControlNetDataPreprocessor,
    SDXLDataPreprocessor,
    StableDiffusion,
    StableDiffusionXL,
    StableDiffusionXLControlNet,
)
from diffengine.models.losses import L2Loss
from diffengine.models.utils import WhiteNoise

try:
    import apex
except ImportError:
    apex = None


class TestFastNormHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="StableDiffusion", module=StableDiffusion)
        MODELS.register_module(
            name="StableDiffusionXL", module=StableDiffusionXL)
        MODELS.register_module(
            name="SDDataPreprocessor", module=SDDataPreprocessor)
        MODELS.register_module(
            name="SDXLDataPreprocessor", module=SDXLDataPreprocessor)
        MODELS.register_module(
            name="StableDiffusionXLControlNet",
            module=StableDiffusionXLControlNet)
        MODELS.register_module(
            name="SDXLControlNetDataPreprocessor",
            module=SDXLControlNetDataPreprocessor)
        MODELS.register_module(name="L2Loss", module=L2Loss)
        MODELS.register_module(name="WhiteNoise", module=WhiteNoise)
        return super().setUp()

    def tearDown(self) -> None:
        MODELS.module_dict.pop("StableDiffusion")
        MODELS.module_dict.pop("StableDiffusionXL")
        MODELS.module_dict.pop("SDDataPreprocessor")
        MODELS.module_dict.pop("SDXLDataPreprocessor")
        MODELS.module_dict.pop("StableDiffusionXLControlNet")
        MODELS.module_dict.pop("SDXLControlNetDataPreprocessor")
        MODELS.module_dict.pop("L2Loss")
        MODELS.module_dict.pop("WhiteNoise")
        return super().tearDown()

    @unittest.skipIf(apex is None, "apex is not installed")
    def test_init(self) -> None:
        FastNormHook()

    @unittest.skipIf(apex is None, "apex is not installed")
    def test_before_train(self) -> None:
        from apex.normalization import FusedLayerNorm

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusion"
        cfg.model.model = "diffusers/tiny-stable-diffusion-torch"
        runner = self.build_runner(cfg)
        hook = FastNormHook(fuse_text_encoder_ln=True)
        assert isinstance(
            runner.model.unet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, nn.LayerNorm)
        assert isinstance(
            runner.model.text_encoder.text_model.encoder.layers[
                0].layer_norm1, nn.LayerNorm)
        # replace norm
        hook.before_train(runner)
        assert isinstance(
            runner.model.unet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, FusedLayerNorm)
        assert isinstance(
            runner.model.text_encoder.text_model.encoder.layers[
                0].layer_norm1, FusedLayerNorm)

        # Test StableDiffusionXL
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusionXL"
        cfg.model.model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        runner = self.build_runner(cfg)
        hook = FastNormHook(fuse_text_encoder_ln=True)
        assert isinstance(
            runner.model.unet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, nn.LayerNorm)
        assert isinstance(
            runner.model.text_encoder_one.text_model.encoder.layers[
                0].layer_norm1, nn.LayerNorm)
        assert isinstance(
            runner.model.text_encoder_two.text_model.encoder.layers[
                0].layer_norm1, nn.LayerNorm)
        # replace norm
        hook.before_train(runner)
        assert isinstance(
            runner.model.unet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, FusedLayerNorm)
        assert isinstance(
            runner.model.text_encoder_one.text_model.encoder.layers[
                0].layer_norm1, FusedLayerNorm)
        assert isinstance(
            runner.model.text_encoder_two.text_model.encoder.layers[
                0].layer_norm1, FusedLayerNorm)

        # Test StableDiffusionXLControlNet
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusionXLControlNet"
        cfg.model.model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        cfg.model.controlnet_model = "hf-internal-testing/tiny-controlnet-sdxl"
        runner = self.build_runner(cfg)
        hook = FastNormHook(fuse_text_encoder_ln=True)
        assert isinstance(
            runner.model.unet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, nn.LayerNorm)
        assert isinstance(
            runner.model.controlnet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, nn.LayerNorm)
        assert isinstance(
            runner.model.text_encoder_one.text_model.encoder.layers[
                0].layer_norm1, nn.LayerNorm)
        assert isinstance(
            runner.model.text_encoder_two.text_model.encoder.layers[
                0].layer_norm1, nn.LayerNorm)
        # replace norm
        hook.before_train(runner)
        assert isinstance(
            runner.model.unet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, FusedLayerNorm)
        assert isinstance(
            runner.model.controlnet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, FusedLayerNorm)
        assert isinstance(
            runner.model.text_encoder_one.text_model.encoder.layers[
                0].layer_norm1, FusedLayerNorm)
        assert isinstance(
            runner.model.text_encoder_two.text_model.encoder.layers[
                0].layer_norm1, FusedLayerNorm)
