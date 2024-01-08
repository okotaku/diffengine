from mmengine.config import read_base

with read_base():
    from .._base_.datasets.fill50k_controlnet_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl_controlnet import *
    from .._base_.schedules.stable_diffusion_3e import *


model.update(transformer_layers_per_block=[0, 0, 1])

train_dataloader.update(batch_size=4)

optim_wrapper.update(
    optimizer=dict(lr=3e-5),
    accumulative_counts=2,
)
