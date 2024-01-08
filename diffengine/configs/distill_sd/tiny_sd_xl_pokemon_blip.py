from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.tiny_sd_xl import *
    from .._base_.schedules.stable_diffusion_xl_50e import *
