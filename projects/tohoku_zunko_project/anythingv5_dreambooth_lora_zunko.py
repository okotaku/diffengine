from mmengine.config import read_base

from diffengine.engine.hooks import PeftSaveHook, VisualizationHook

with read_base():
    from diffengine.configs._base_.default_runtime import *
    from diffengine.configs._base_.schedules.stable_diffusion_1k import *

    from ._base_.anythingv5_lora import *
    from ._base_.zunko_dreambooth import *

train_dataloader.update(
    dataset=dict(
        class_image_config=dict(model=model.model),
        instance_prompt="1girl, sks"))

custom_hooks = [
    dict(
        type=VisualizationHook,
        prompt=["1girl, sks, in a bucket"] * 4,
        by_epoch=False,
        interval=100),
    dict(type=PeftSaveHook),
]
