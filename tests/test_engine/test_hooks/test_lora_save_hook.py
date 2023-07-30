import copy
import gc
import os
import os.path as osp
from pathlib import Path

import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase

from diffengine.engine.hooks import LoRASaveHook
from diffengine.models.editors import StableDiffusion
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
        MODELS.register_module(name='StableDiffusion', module=StableDiffusion)
        MODELS.register_module(name='L2Loss', module=L2Loss)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop('DummyWrapper')
        MODELS.module_dict.pop('StableDiffusion')
        MODELS.module_dict.pop('L2Loss')
        return super().tearDown()

    def test_init(self):
        LoRASaveHook()

    def test_before_save_checkpoint(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = 'StableDiffusion'
        cfg.model.lora_config = dict(rank=4)
        runner = self.build_runner(cfg)
        checkpoint = dict(
            state_dict=StableDiffusion(lora_config=dict(rank=4)).state_dict())
        hook = LoRASaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f'step{runner.iter}',
                     'pytorch_lora_weights.bin')).exists
        os.remove(
            osp.join(runner.work_dir, f'step{runner.iter}',
                     'pytorch_lora_weights.bin'))

        for key in checkpoint['state_dict'].keys():
            assert key.startswith(tuple(['unet', 'text_encoder']))
        del runner, checkpoint
        gc.collect()

    def test_before_save_checkpoint_text_encoder(self):
        # with text encoder
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = 'StableDiffusion'
        cfg.model.lora_config = dict(rank=4)
        cfg.model.finetune_text_encoder = True
        runner = self.build_runner(cfg)
        checkpoint = dict(
            state_dict=StableDiffusion(
                lora_config=dict(
                    rank=4), finetune_text_encoder=True).state_dict())
        hook = LoRASaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f'step{runner.iter}',
                     'pytorch_lora_weights.bin')).exists
        os.remove(
            osp.join(runner.work_dir, f'step{runner.iter}',
                     'pytorch_lora_weights.bin'))

        for key in checkpoint['state_dict'].keys():
            assert key.startswith(tuple(['unet', 'text_encoder']))
        del runner, checkpoint
        gc.collect()
