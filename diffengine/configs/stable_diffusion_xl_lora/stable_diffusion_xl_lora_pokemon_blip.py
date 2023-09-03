from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl_lora import *
    from .._base_.schedules.stable_diffusion_50e import *

from diffengine.engine.hooks import LoRASaveHook, VisualizationHook

custom_hooks = [
    dict(type=VisualizationHook, prompt=['yoda pokemon'] * 4),
    dict(type=LoRASaveHook),
]
