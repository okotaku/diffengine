from mmengine.config import read_base

with read_base():
    from .._base_.datasets.dog_dreambooth import *
    from .._base_.default_runtime import *
    from .._base_.models.small_sd_lora import *
    from .._base_.schedules.stable_diffusion_1k import *


train_dataloader.update(
    dataset=dict(class_image_config=dict(model=model.model)))
