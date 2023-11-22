import copy

import torch
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase, assert_allclose
from mmengine.testing.runner_test_case import ToyModel
from torch import nn

from diffengine.engine.hooks import LCMEMAUpdateHook


class DummyWrapper(BaseModel):

    def __init__(self, model) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            model = MODELS.build(model)
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ToyModel2(ToyModel):

    def __init__(self) -> None:
        super().__init__()
        ema_cfg = dict(type="ExponentialMovingAverage", momentum=0.05)
        self.unet = nn.Linear(2, 1)
        self.target_unet = MODELS.build(
                ema_cfg, default_args=dict(model=self.unet))

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
        LCMEMAUpdateHook()

    def _get_ema_hook(self, runner):
        for hook in runner.hooks:
            if isinstance(hook, LCMEMAUpdateHook):
                return hook
        return None

    def test_after_train_iter(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "ToyModel2"
        cfg.custom_hooks = [dict(type="LCMEMAUpdateHook")]
        runner = self.build_runner(cfg)
        ema_hook = self._get_ema_hook(runner)

        ema_hook.before_run(runner)
        ema_hook.before_train(runner)

        src_model = ema_hook.src_model
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
