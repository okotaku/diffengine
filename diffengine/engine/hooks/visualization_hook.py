from typing import List, Optional, Union

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
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
        height (`int`, *optional*, defaults to
            `self.unet.config.sample_size * self.vae_scale_factor`):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to
            `self.unet.config.sample_size * self.vae_scale_factor`):
            The width in pixels of the generated image.
    """
    priority = 'NORMAL'

    def __init__(self,
                 prompt: List[str],
                 interval: int = 1,
                 by_epoch: bool = True,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        self.interval = interval
        self.by_epoch = by_epoch
        self.height = height
        self.width = width

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
            model = runner.model
            if is_model_wrapper(model):
                model = model.module
            images = model.infer(
                self.prompt,
                height=self.height,
                width=self.width,
                **self.kwargs)
            for i, image in enumerate(images):
                runner.visualizer.add_image(
                    f'image{i}_step', image, step=runner.iter)

    def after_train_epoch(self, runner) -> None:
        """
        Args:
            runner (Runner): The runner of the training process.
        """
        if self.by_epoch:
            model = runner.model
            if is_model_wrapper(model):
                model = model.module
            images = model.infer(
                self.prompt,
                height=self.height,
                width=self.width,
                **self.kwargs)
            for i, image in enumerate(images):
                runner.visualizer.add_image(
                    f'image{i}_step', image, step=runner.epoch)
