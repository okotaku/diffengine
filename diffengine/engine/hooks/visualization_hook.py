from typing import Optional

from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS


@HOOKS.register_module()
class VisualizationHook(Hook):
    """Basic hook that invoke visualizers after train epoch.

    Args:
    ----
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

    priority = "NORMAL"

    def __init__(self,
                 prompt: list[str],
                 interval: int = 1,
                 height: int | None = None,
                 width: int | None = None,
                 *,
                 by_epoch: bool = True,
                 **kwargs) -> None:
        self.prompt = prompt
        self.kwargs = kwargs
        self.interval = interval
        self.by_epoch = by_epoch
        self.height = height
        self.width = width

    def after_train_iter(
            self,
            runner,
            batch_idx: int,
            data_batch: DATA_BATCH = None,  # noqa
            outputs: Optional[dict] = None) -> None:  # noqa
        """After train iter hook.

        Args:
        ----
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch.
            data_batch (DATA_BATCH, optional): The current data batch.
            outputs (dict, optional): The outputs of the current batch.
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
                    f"image{i}_step", image, step=runner.iter)

    def after_train_epoch(self, runner) -> None:
        """After train epoch hook.

        Args:
        ----
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
                    f"image{i}_step", image, step=runner.epoch)
