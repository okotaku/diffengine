import copy
import re

import pytest
import torch
from mmengine.model import BaseModel, ExponentialMovingAverage
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase, assert_allclose
from mmengine.testing.runner_test_case import ToyModel
from torch import nn

from diffengine.engine.hooks import UnetEMAHook


class DummyWrapper(BaseModel):

    def __init__(self, model):
        super().__init__()
        if not isinstance(model, nn.Module):
            model = MODELS.build(model)
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ToyModel2(ToyModel):

    def __init__(self):
        super().__init__()
        self.unet = nn.Linear(2, 1)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class TestEMAHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="DummyWrapper", module=DummyWrapper)
        MODELS.register_module(name="ToyModel2", module=ToyModel2)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop("DummyWrapper")
        MODELS.module_dict.pop("ToyModel2")
        return super().tearDown()

    def test_init(self):
        UnetEMAHook()

        with pytest.raises(AssertionError, match="`begin_iter` must"):
            UnetEMAHook(begin_iter=-1)

        with pytest.raises(AssertionError, match="`begin_epoch` must"):
            UnetEMAHook(begin_epoch=-1)

        with pytest.raises(
                AssertionError, match="`begin_iter` and `begin_epoch`"):
            UnetEMAHook(begin_iter=1, begin_epoch=1)

    def _get_ema_hook(self, runner):
        for hook in runner.hooks:
            if isinstance(hook, UnetEMAHook):
                return hook
        return None

    def test_after_train_iter(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "ToyModel2"
        cfg.custom_hooks = [{"type": "UnetEMAHook"}]
        runner = self.build_runner(cfg)
        ema_hook = self._get_ema_hook(runner)

        ema_hook = self._get_ema_hook(runner)
        ema_hook.before_run(runner)
        ema_hook.before_train(runner)

        src_model = runner.model.unet
        ema_model = ema_hook.ema_model

        with torch.no_grad():
            for parameter in src_model.parameters():
                parameter.data.copy_(torch.randn(parameter.shape))

        ema_hook.after_train_iter(runner, 1)
        for src, ema in zip(
                src_model.parameters(), ema_model.parameters(), strict=True):
            assert_allclose(src.data, ema.data)

        with torch.no_grad():
            for parameter in src_model.parameters():
                parameter.data.copy_(torch.randn(parameter.shape))

        ema_hook.after_train_iter(runner, 1)

        for src, ema in zip(
                src_model.parameters(), ema_model.parameters(), strict=True):
            assert not (src.data == ema.data).all()

    def test_before_save_checkpoint(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "ToyModel2"
        runner = self.build_runner(cfg)
        checkpoint = {"state_dict": ToyModel2().state_dict()}
        ema_hook = UnetEMAHook()
        ema_hook.before_run(runner)
        ema_hook.before_train(runner)

        ori_checkpoint = copy.deepcopy(checkpoint)
        ema_hook.before_save_checkpoint(runner, checkpoint)

        for key in ori_checkpoint["state_dict"]:
            if key.startswith("unet."):
                assert_allclose(
                    ori_checkpoint["state_dict"][key].cpu(),
                    checkpoint["ema_state_dict"][f"module.{key[5:]}"].cpu())

                assert_allclose(
                    ema_hook.ema_model.state_dict()[f"module.{key[5:]}"].cpu(),
                    checkpoint["state_dict"][key].cpu())

    def test_after_load_checkpoint(self):
        # Test load a checkpoint without ema_state_dict.
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "ToyModel2"
        runner = self.build_runner(cfg)
        checkpoint = {"state_dict": ToyModel2().state_dict()}
        ema_hook = UnetEMAHook()
        ema_hook.before_run(runner)
        ema_hook.before_train(runner)
        ema_hook.after_load_checkpoint(runner, checkpoint)

        for key in checkpoint["state_dict"]:
            if key.startswith("unet."):
                assert_allclose(
                    checkpoint["state_dict"][key].cpu(),
                    ema_hook.ema_model.state_dict()[f"module.{key[5:]}"].cpu())

        # Test a warning should be raised when resuming from a checkpoint
        # without `ema_state_dict`
        runner._resume = True
        ema_hook.after_load_checkpoint(runner, checkpoint)
        with self.assertLogs(runner.logger, level="WARNING") as cm:
            ema_hook.after_load_checkpoint(runner, checkpoint)
            assert re.search("There is no `ema_state_dict`", cm.records[0].msg)

        # Check the weight of state_dict and ema_state_dict have been swapped.
        # when runner._resume is True
        runner._resume = True
        checkpoint = {
            "state_dict":
            ToyModel2().state_dict(),
            "ema_state_dict":
            ExponentialMovingAverage(ToyModel2().unet).state_dict(),
        }
        ori_checkpoint = copy.deepcopy(checkpoint)
        ema_hook.after_load_checkpoint(runner, checkpoint)
        for key in ori_checkpoint["state_dict"]:
            if key.startswith("unet."):
                assert_allclose(
                    ori_checkpoint["state_dict"][key].cpu(),
                    ema_hook.ema_model.state_dict()[f"module.{key[5:]}"].cpu())

        runner._resume = False
        ema_hook.after_load_checkpoint(runner, checkpoint)
