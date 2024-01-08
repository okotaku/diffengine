from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_kandinsky_decoder import *
    from .._base_.default_runtime import *
    from .._base_.models.kandinsky_v22_decoder import *
    from .._base_.schedules.stable_diffusion_50e import *
