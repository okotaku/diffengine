from mmengine.config import read_base

with read_base():
    from .._base_.datasets.cat_waterpainting_dreambooth_pixart_1024 import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_1024_lora import *
    from .._base_.schedules.stable_diffusion_1k import *

model.update(transformer_lora_config=dict(r=64, lora_alpha=64))

train_dataloader.update(
    dataset=dict(class_image_config=dict(model={{_base_.model.model}})))

optim_wrapper.update(
    dtype="bfloat16",
    optimizer=dict(lr=1e-4))
