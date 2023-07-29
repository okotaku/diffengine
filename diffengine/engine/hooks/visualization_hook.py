from typing import List

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class VisualizationHook(Hook):
    """Basic hook that invoke visualizers after train epoch.

    Args:
        prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
    """
    priority = 'NORMAL'

    def __init__(self, prompt: List[str]):
        self.prompt = prompt

    def after_train_epoch(self, runner) -> None:
        images = runner.model.infer(self.prompt)
        for i, image in enumerate(images):
            runner.visualizer.add_image(
                f'image{i}_step', image, step=runner.epoch)
