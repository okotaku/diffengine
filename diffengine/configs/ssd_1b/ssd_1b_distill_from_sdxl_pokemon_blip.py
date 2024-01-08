from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.distill_ssd_1b_from_sdxl import *
    from .._base_.schedules.stable_diffusion_xl_50e import *
