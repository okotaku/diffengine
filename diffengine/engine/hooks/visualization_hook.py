from typing import List, Optional, Union

from mmengine.hooks import Hook
from mmengine.registry import HOOKS

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class VisualizationHook(Hook):
    """Basic hook that invoke visualizers after train epoch.

    Args:
        prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
        interval (int): Visualization interval (every k iterations).
            Defaults to 1.
        by_epoch (bool): Whether to visualize by epoch. Defaults to True.
    """
    priority = 'NORMAL'

    def __init__(self,
                 prompt: List[str],
                 interval: int = 1,
                 by_epoch: bool = True):
        self.prompt = prompt
        self.interval = interval
        self.by_epoch = by_epoch

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """
        Args:
            runner (Runner): The runner of the training process.
        """
        if self.by_epoch:
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            images = runner.model.infer(self.prompt)
            for i, image in enumerate(images):
                runner.visualizer.add_image(
                    f'image{i}_step', image, step=runner.iter)

    def after_train_epoch(self, runner) -> None:
        """
        Args:
            runner (Runner): The runner of the training process.
        """
        if self.by_epoch:
            images = runner.model.infer(self.prompt)
            for i, image in enumerate(images):
                runner.visualizer.add_image(
                    f'image{i}_step', image, step=runner.epoch)
