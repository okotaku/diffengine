from mmengine.config import read_base

with read_base():
    from .._base_.datasets.dog_dreambooth_pixart_1024 import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_1024_lora import *
    from .._base_.schedules.stable_diffusion_500 import *

train_dataloader.update(
    dataset=dict(class_image_config=dict(model=model.model)))

optim_wrapper.update(
    dtype="bfloat16",
    optimizer=dict(lr=1e-4))
