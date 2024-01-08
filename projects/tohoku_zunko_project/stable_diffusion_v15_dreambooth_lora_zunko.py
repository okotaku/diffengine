from mmengine.config import read_base

with read_base():
    from diffengine.configs._base_.default_runtime import *
    from diffengine.configs._base_.models.stable_diffusion_v15_lora import *
    from diffengine.configs._base_.schedules.stable_diffusion_1k import *

    from ._base_.zunko_dreambooth import *

train_dataloader.update(
    dataset=dict(class_image_config=dict(model=model.model)))
