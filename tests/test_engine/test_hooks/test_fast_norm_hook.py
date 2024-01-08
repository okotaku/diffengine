import copy
import unittest

from diffusers import AutoencoderKL, Transformer2DModel
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from mmengine.testing.runner_test_case import ToyModel
from torch import nn
from transformers import T5EncoderModel

from diffengine.engine.hooks import FastNormHook
from diffengine.models.utils import TimeSteps, WhiteNoise

try:
    import apex
except ImportError:
    apex = None



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

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class TestFastNormHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="WhiteNoise", module=WhiteNoise)
        MODELS.register_module(name="TimeSteps", module=TimeSteps)
        MODELS.register_module(name="ToyModelPixArt", module=ToyModelPixArt)
        return super().setUp()

    def tearDown(self) -> None:
        MODELS.module_dict.pop("WhiteNoise")
        MODELS.module_dict.pop("TimeSteps")
        MODELS.module_dict.pop("ToyModelPixArt")
        return super().tearDown()

    @unittest.skipIf(apex is None, "apex is not installed")
    def test_init(self) -> None:
        FastNormHook()

    @unittest.skipIf(apex is None, "apex is not installed")
    def test_before_train(self) -> None:
        from apex.contrib.group_norm import GroupNorm
        from apex.normalization import FusedLayerNorm

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/sd.py").model
        runner = self.build_runner(cfg)
        hook = FastNormHook(fuse_text_encoder_ln=True, fuse_gn=True)
        assert isinstance(
            runner.model.unet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, nn.LayerNorm)
        assert isinstance(
            runner.model.unet.down_blocks[0].resnets[0].norm2, nn.GroupNorm)
        assert isinstance(
            runner.model.text_encoder.text_model.encoder.layers[
                0].layer_norm1, nn.LayerNorm)
        # replace norm
        hook.before_train(runner)
        assert isinstance(
            runner.model.unet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, FusedLayerNorm)
        assert isinstance(
            runner.model.unet.down_blocks[0].resnets[0].norm2, GroupNorm)
        assert isinstance(
            runner.model.text_encoder.text_model.encoder.layers[
                0].layer_norm1, FusedLayerNorm)

        # Test StableDiffusionXL
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/sdxl.py").model
        runner = self.build_runner(cfg)
        hook = FastNormHook(fuse_text_encoder_ln=True, fuse_gn=True)
        assert isinstance(
            runner.model.unet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, nn.LayerNorm)
        assert isinstance(
            runner.model.unet.down_blocks[0].resnets[0].norm2, nn.GroupNorm)
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
            runner.model.unet.down_blocks[0].resnets[0].norm2, GroupNorm)
        assert isinstance(
            runner.model.text_encoder_one.text_model.encoder.layers[
                0].layer_norm1, FusedLayerNorm)
        assert isinstance(
            runner.model.text_encoder_two.text_model.encoder.layers[
                0].layer_norm1, FusedLayerNorm)

        # Test StableDiffusionXLControlNet
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/sdxlcn.py").model
        runner = self.build_runner(cfg)
        hook = FastNormHook(fuse_text_encoder_ln=True, fuse_gn=True)
        assert isinstance(
            runner.model.unet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, nn.LayerNorm)
        assert isinstance(
            runner.model.controlnet.down_blocks[
                1].attentions[0].transformer_blocks[0].norm1, nn.LayerNorm)
        assert isinstance(
            runner.model.unet.down_blocks[0].resnets[0].norm2, nn.GroupNorm)
        assert isinstance(
            runner.model.controlnet.down_blocks[0].resnets[0].norm2,
            nn.GroupNorm)
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
            runner.model.unet.down_blocks[0].resnets[0].norm2, GroupNorm)
        assert isinstance(
            runner.model.controlnet.down_blocks[0].resnets[0].norm2,
            GroupNorm)
        assert isinstance(
            runner.model.text_encoder_one.text_model.encoder.layers[
                0].layer_norm1, FusedLayerNorm)
        assert isinstance(
            runner.model.text_encoder_two.text_model.encoder.layers[
                0].layer_norm1, FusedLayerNorm)

        # test PixArt
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/pixart.py").model
        runner = self.build_runner(cfg)
        hook = FastNormHook(fuse_text_encoder_ln=True, fuse_gn=True)
        assert isinstance(
            runner.model.transformer.transformer_blocks[
                0].norm1, nn.LayerNorm)
        # replace norm
        hook.before_train(runner)
        assert isinstance(
            runner.model.transformer.transformer_blocks[
                0].norm1, FusedLayerNorm)
