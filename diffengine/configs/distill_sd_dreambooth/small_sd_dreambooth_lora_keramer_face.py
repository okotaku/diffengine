from mmengine.config import read_base

from diffengine.engine.hooks import PeftSaveHook, VisualizationHook

with read_base():
    from .._base_.datasets.keramer_face_dreambooth import *
    from .._base_.default_runtime import *
    from .._base_.models.small_sd_lora import *
    from .._base_.schedules.stable_diffusion_1k import *


train_dataloader.update(
    dataset=dict(
        class_image_config=dict(mode=model.model),
        instance_prompt="Portrait photo of a sks person",
        class_prompt="Portrait photo of a person"))

custom_hooks = [
    dict(
        type=VisualizationHook,
        prompt=["Portrait photo of a sks person in suits"] * 4,
        by_epoch=False,
        interval=100),
    dict(type=PeftSaveHook),
]
