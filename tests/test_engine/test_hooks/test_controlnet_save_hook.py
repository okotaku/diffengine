import copy
import os.path as osp
import shutil
from pathlib import Path

import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase

from diffengine.engine.hooks import ControlNetSaveHook
from diffengine.models.editors import (SDControlNetDataPreprocessor,
                                       StableDiffusionControlNet)
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
        MODELS.register_module(name='DummyWrapper', module=DummyWrapper)
        MODELS.register_module(
            name='StableDiffusionControlNet', module=StableDiffusionControlNet)
        MODELS.register_module(
            name='SDControlNetDataPreprocessor',
            module=SDControlNetDataPreprocessor)
        MODELS.register_module(name='L2Loss', module=L2Loss)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop('DummyWrapper')
        MODELS.module_dict.pop('StableDiffusionControlNet')
        MODELS.module_dict.pop('SDControlNetDataPreprocessor')
        MODELS.module_dict.pop('L2Loss')
        return super().tearDown()

    def test_init(self):
        ControlNetSaveHook()

    def test_before_save_checkpoint(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = 'StableDiffusionControlNet'
        cfg.model.model = 'diffusers/tiny-stable-diffusion-torch'
        runner = self.build_runner(cfg)
        checkpoint = dict(
            state_dict=StableDiffusionControlNet(
                model='diffusers/tiny-stable-diffusion-torch').state_dict())
        hook = ControlNetSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f'step{runner.iter}', 'controlnet',
                     'diffusion_pytorch_model.safetensors')).exists()
        shutil.rmtree(osp.join(runner.work_dir), ignore_errors=True)

        for key in checkpoint['state_dict'].keys():
            assert key.startswith('controlnet')
