import copy
import os.path as osp
import shutil
from pathlib import Path

from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from torch import nn

from diffengine.engine.hooks import T2IAdapterSaveHook
from diffengine.models.utils import CubicSamplingTimeSteps, WhiteNoise


class DummyWrapper(BaseModel):

    def __init__(self, model) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            model = MODELS.build(model)
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class TestT2IAdapterSaveHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="DummyWrapper", module=DummyWrapper)
        MODELS.register_module(name="WhiteNoise", module=WhiteNoise)
        MODELS.register_module(name="CubicSamplingTimeSteps",
                               module=CubicSamplingTimeSteps)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop("DummyWrapper")
        MODELS.module_dict.pop("WhiteNoise")
        MODELS.module_dict.pop("CubicSamplingTimeSteps")
        return super().tearDown()

    def test_init(self):
        T2IAdapterSaveHook()

    def test_before_save_checkpoint(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/sdxl_t2iadapter.py").model
        runner = self.build_runner(cfg)
        checkpoint = dict(
            state_dict=MODELS.build(cfg.model).state_dict())
        hook = T2IAdapterSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/adapter",
                     "diffusion_pytorch_model.safetensors")).exists()
        shutil.rmtree(
            osp.join(runner.work_dir, f"step{runner.iter}"))

        assert len(checkpoint["state_dict"]) > 0
        for key in checkpoint["state_dict"]:
            assert key.startswith("adapter")
