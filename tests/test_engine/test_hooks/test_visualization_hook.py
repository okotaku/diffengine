import copy
from unittest.mock import MagicMock

from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import EpochBasedTrainLoop
from mmengine.testing import RunnerTestCase

from diffengine.engine.hooks import VisualizationHook
from diffengine.models.utils import TimeSteps, WhiteNoise


class TestVisualizationHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="WhiteNoise", module=WhiteNoise)
        MODELS.register_module(name="TimeSteps", module=TimeSteps)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop("WhiteNoise")
        MODELS.module_dict.pop("TimeSteps")
        return super().tearDown()

    def test_after_train_epoch(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        hook = VisualizationHook(prompt=["a dog"])
        hook.after_train_epoch(runner)

    def test_after_train_epoch_with_condition(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        hook = VisualizationHook(
            prompt=["a dog"], condition_image=["testdata/color.jpg"])
        hook.after_train_epoch(runner)

    def test_after_train_epoch_with_example_iamge(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        hook = VisualizationHook(
            prompt=["a dog"], example_image=["testdata/color.jpg"])
        hook.after_train_epoch(runner)

    def test_after_train_iter(self):
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.train_cfg.max_iters = 100
        cfg.model = Config.fromfile("tests/configs/sd.py").model
        runner = self.build_runner(cfg)
        hook = VisualizationHook(prompt=["a dog"], by_epoch=False)
        for i in range(10):
            hook.after_train_iter(runner, i)
            runner.train_loop._iter += 1

    def test_after_train_iter_with_condition(self):
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.train_cfg.max_iters = 100
        cfg.model = Config.fromfile("tests/configs/sdcn.py").model
        runner = self.build_runner(cfg)
        hook = VisualizationHook(
            prompt=["a dog"],
            condition_image=["tests/testdata/cond.jpg"],
            height=64,
            width=64,
            by_epoch=False)
        for i in range(10):
            hook.after_train_iter(runner, i)
            runner.train_loop._iter += 1
