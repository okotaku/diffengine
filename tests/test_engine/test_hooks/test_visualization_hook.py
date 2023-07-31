import copy
from unittest.mock import MagicMock

from mmengine.registry import MODELS
from mmengine.runner import EpochBasedTrainLoop
from mmengine.testing import RunnerTestCase

from diffengine.engine.hooks import VisualizationHook
from diffengine.models.editors import StableDiffusion
from diffengine.models.losses import L2Loss


class TestVisualizationHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name='StableDiffusion', module=StableDiffusion)
        MODELS.register_module(name='L2Loss', module=L2Loss)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop('StableDiffusion')
        MODELS.module_dict.pop('L2Loss')
        return super().tearDown()

    def test_after_train_epoch(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        hook = VisualizationHook(prompt=['a dog'])
        hook.after_train_epoch(runner)

    def test_after_train_iter(self):
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.train_cfg.max_iters = 100
        cfg.model.type = 'StableDiffusion'
        cfg.model.model = 'diffusers/tiny-stable-diffusion-torch'
        runner = self.build_runner(cfg)
        hook = VisualizationHook(prompt=['a dog'], by_epoch=False)
        for i in range(10):
            hook.after_train_iter(runner, i)
            runner.train_loop._iter += 1
