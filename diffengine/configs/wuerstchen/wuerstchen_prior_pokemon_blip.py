from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_wuerstchen import *
    from .._base_.default_runtime import *
    from .._base_.models.wuerstchen_prior import *
    from .._base_.schedules.stable_diffusion_50e import *

optim_wrapper.update(
    optimizer=dict(lr=1e-5),
    accumulative_counts=4)  # update every four times
