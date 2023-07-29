from unittest import TestCase
from unittest.mock import MagicMock

from mmengine.runner import EpochBasedTrainLoop

from diffengine.engine.hooks import VisualizationHook


class TestVisualizationHook(TestCase):

    def test_after_train_epoch(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        hook = VisualizationHook(prompt=['a dog'])
        hook.after_train_epoch(runner)
