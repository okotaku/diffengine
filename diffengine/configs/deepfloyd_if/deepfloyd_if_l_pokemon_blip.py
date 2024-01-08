from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_if import *
    from .._base_.default_runtime import *
    from .._base_.models.deepfloyd_if_l import *
    from .._base_.schedules.stable_diffusion_50e import *


optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-2),
    clip_grad=dict(max_norm=1.0))
