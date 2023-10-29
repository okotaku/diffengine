import copy
import unittest

from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from torch import nn

from diffengine.engine.hooks import FastNormHook
from diffengine.models.editors import (
    SDDataPreprocessor,
    SDXLDataPreprocessor,
    StableDiffusion,
    StableDiffusionXL,
)
from diffengine.models.losses import L2Loss

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
        MODELS.register_module(name="L2Loss", module=L2Loss)
        return super().setUp()

    def tearDown(self) -> None:
        MODELS.module_dict.pop("StableDiffusion")
        MODELS.module_dict.pop("StableDiffusionXL")
        MODELS.module_dict.pop("SDDataPreprocessor")
        MODELS.module_dict.pop("SDXLDataPreprocessor")
        MODELS.module_dict.pop("L2Loss")
        return super().tearDown()

    def test_init(self) -> None:
        FastNormHook()

    @unittest.skipIf(apex is None, "apex is not installed")
    def test_before_train(self) -> None:
        from apex.normalization import FusedLayerNorm

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusion"
        cfg.model.model = "diffusers/tiny-stable-diffusion-torch"
        runner = self.build_runner(cfg)
        hook = FastNormHook()
        assert isinstance(
            runner.model.unet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, nn.LayerNorm)
        # replace norm
        hook.before_train(runner)
        assert isinstance(
            runner.model.unet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, FusedLayerNorm)

        # Test StableDiffusionXL
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusionXL"
        cfg.model.model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        runner = self.build_runner(cfg)
        hook = FastNormHook()
        assert isinstance(
            runner.model.unet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, nn.LayerNorm)
        # replace norm
        hook.before_train(runner)
        assert isinstance(
            runner.model.unet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, FusedLayerNorm)
