from mmengine.config import read_base

with read_base():
    from .._base_.datasets.fill50k_controlnet_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl_controlnet import *
    from .._base_.schedules.stable_diffusion_1e import *


optim_wrapper.update(
    optimizer=dict(lr=1e-5),
    accumulative_counts=2,
)
