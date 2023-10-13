import copy
import os
import os.path as osp
from pathlib import Path

from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from torch import nn

from diffengine.engine.hooks import IPAdapterSaveHook
from diffengine.models.editors import IPAdapterXL, IPAdapterXLDataPreprocessor
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
        MODELS.register_module(name="IPAdapterXL", module=IPAdapterXL)
        MODELS.register_module(
            name="IPAdapterXLDataPreprocessor",
            module=IPAdapterXLDataPreprocessor)
        MODELS.register_module(name="L2Loss", module=L2Loss)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop("DummyWrapper")
        MODELS.module_dict.pop("IPAdapterXL")
        MODELS.module_dict.pop("IPAdapterXLDataPreprocessor")
        MODELS.module_dict.pop("L2Loss")
        return super().tearDown()

    def test_init(self):
        IPAdapterSaveHook()

    def test_before_save_checkpoint(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "IPAdapterXL"
        cfg.model.model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        cfg.model.image_encoder = 'hf-internal-testing/unidiffuser-diffusers-test'  # noqa
        runner = self.build_runner(cfg)
        checkpoint = {
            "state_dict":
            IPAdapterXL(
                model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
                image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
            ).state_dict(),
        }
        hook = IPAdapterSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}",
                     "pytorch_lora_weights.safetensors")).exists()
        os.remove(
            osp.join(runner.work_dir, f"step{runner.iter}",
                     "pytorch_lora_weights.safetensors"))

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/image_projection",
                     "diffusion_pytorch_model.safetensors")).exists()
        os.remove(
            osp.join(runner.work_dir, f"step{runner.iter}/image_projection",
                     "diffusion_pytorch_model.safetensors"))

        for key in checkpoint["state_dict"]:
            assert key.startswith(("unet", "image_projection"))
            assert ".processor." in key or key.startswith("image_projection")
