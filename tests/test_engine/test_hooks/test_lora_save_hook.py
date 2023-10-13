import copy
import os
import os.path as osp
from pathlib import Path

from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from torch import nn

from diffengine.engine.hooks import LoRASaveHook
from diffengine.models.editors import (
    SDDataPreprocessor,
    SDXLDataPreprocessor,
    StableDiffusion,
    StableDiffusionXL,
)
from diffengine.models.losses import L2Loss


class DummyWrapper(BaseModel):

    def __init__(self, model):
        super().__init__()
        if not isinstance(model, nn.Module):
            model = MODELS.build(model)
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class TestLoRASaveHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="DummyWrapper", module=DummyWrapper)
        MODELS.register_module(name="StableDiffusion", module=StableDiffusion)
        MODELS.register_module(
            name="StableDiffusionXL", module=StableDiffusionXL)
        MODELS.register_module(
            name="SDDataPreprocessor", module=SDDataPreprocessor)
        MODELS.register_module(
            name="SDXLDataPreprocessor", module=SDXLDataPreprocessor)
        MODELS.register_module(name="L2Loss", module=L2Loss)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop("DummyWrapper")
        MODELS.module_dict.pop("StableDiffusion")
        MODELS.module_dict.pop("StableDiffusionXL")
        MODELS.module_dict.pop("SDDataPreprocessor")
        MODELS.module_dict.pop("SDXLDataPreprocessor")
        MODELS.module_dict.pop("L2Loss")
        return super().tearDown()

    def test_init(self):
        LoRASaveHook()

    def test_before_save_checkpoint(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusion"
        cfg.model.lora_config = {"rank": 4}
        cfg.model.model = "diffusers/tiny-stable-diffusion-torch"
        runner = self.build_runner(cfg)
        checkpoint = {
            "state_dict":
            StableDiffusion(
                model="diffusers/tiny-stable-diffusion-torch",
                lora_config={
                    "rank": 4,
                }).state_dict(),
        }
        hook = LoRASaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}",
                     "pytorch_lora_weights.safetensors")).exists()
        os.remove(
            osp.join(runner.work_dir, f"step{runner.iter}",
                     "pytorch_lora_weights.safetensors"))

        for key in checkpoint["state_dict"]:
            assert key.startswith(("unet", "text_encoder"))

    def test_before_save_checkpoint_text_encoder(self):
        # with text encoder
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusion"
        cfg.model.lora_config = {"rank": 4}
        cfg.model.finetune_text_encoder = True
        cfg.model.model = "diffusers/tiny-stable-diffusion-torch"
        runner = self.build_runner(cfg)
        checkpoint = {
            "state_dict":
            StableDiffusion(
                model="diffusers/tiny-stable-diffusion-torch",
                lora_config={
                    "rank": 4,
                },
                finetune_text_encoder=True).state_dict(),
        }
        hook = LoRASaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}",
                     "pytorch_lora_weights.safetensors")).exists()
        os.remove(
            osp.join(runner.work_dir, f"step{runner.iter}",
                     "pytorch_lora_weights.safetensors"))

        for key in checkpoint["state_dict"]:
            assert key.startswith(("unet", "text_encoder"))

        # sdxl
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusionXL"
        cfg.model.lora_config = {"rank": 4}
        cfg.model.finetune_text_encoder = True
        cfg.model.model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        runner = self.build_runner(cfg)
        checkpoint = {
            "state_dict":
            StableDiffusionXL(
                model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
                lora_config={
                    "rank": 4,
                },
                finetune_text_encoder=True).state_dict(),
        }
        hook = LoRASaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}",
                     "pytorch_lora_weights.safetensors")).exists()
        os.remove(
            osp.join(runner.work_dir, f"step{runner.iter}",
                     "pytorch_lora_weights.safetensors"))

        for key in checkpoint["state_dict"]:
            assert key.startswith(("unet", "text_encoder"))
