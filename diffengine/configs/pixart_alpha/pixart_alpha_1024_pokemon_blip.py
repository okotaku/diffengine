from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_pixart import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_1024 import *
    from .._base_.schedules.stable_diffusion_50e import *

optim_wrapper.update(
    dtype="bfloat16",
    optimizer=dict(lr=2e-6, weight_decay=3e-2),
    clip_grad=dict(max_norm=0.01))
