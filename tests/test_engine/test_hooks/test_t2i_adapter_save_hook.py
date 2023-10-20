import copy
import os
import os.path as osp
from pathlib import Path

from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from torch import nn

from diffengine.engine.hooks import T2IAdapterSaveHook
from diffengine.models.editors import (
    SDXLControlNetDataPreprocessor,
    StableDiffusionXLT2IAdapter,
)
from diffengine.models.losses import L2Loss


class DummyWrapper(BaseModel):

    def __init__(self, model) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            model = MODELS.build(model)
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class TestLoRASaveHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="DummyWrapper", module=DummyWrapper)
        MODELS.register_module(
            name="StableDiffusionXLT2IAdapter",
            module=StableDiffusionXLT2IAdapter)
        MODELS.register_module(
            name="SDXLControlNetDataPreprocessor",
            module=SDXLControlNetDataPreprocessor)
        MODELS.register_module(name="L2Loss", module=L2Loss)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop("DummyWrapper")
        MODELS.module_dict.pop("StableDiffusionXLT2IAdapter")
        MODELS.module_dict.pop("SDXLControlNetDataPreprocessor")
        MODELS.module_dict.pop("L2Loss")
        return super().tearDown()

    def test_init(self):
        T2IAdapterSaveHook()

    def test_before_save_checkpoint(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusionXLT2IAdapter"
        cfg.model.model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        cfg.model.adapter_model = "hf-internal-testing/tiny-adapter"
        runner = self.build_runner(cfg)
        checkpoint = dict(
            state_dict=StableDiffusionXLT2IAdapter(
                model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
                adapter_model="hf-internal-testing/tiny-adapter").state_dict())
        hook = T2IAdapterSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/adapter",
                     "diffusion_pytorch_model.safetensors")).exists()
        os.remove(
            osp.join(runner.work_dir, f"step{runner.iter}/adapter",
                     "diffusion_pytorch_model.safetensors"))

        assert len(checkpoint["state_dict"]) > 0
        for key in checkpoint["state_dict"]:
            assert key.startswith("adapter")
