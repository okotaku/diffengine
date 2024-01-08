from mmengine.config import read_base

from diffengine.engine.hooks import PeftSaveHook, VisualizationHook

with read_base():
    from .._base_.datasets.pokemon_blip_pixart import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_1024_lora import *
    from .._base_.schedules.stable_diffusion_50e import *

optim_wrapper.update(
    dtype="bfloat16")

custom_hooks = [
    dict(
        type=VisualizationHook,
        prompt=["yoda pokemon"] * 4,
        height=1024,
        width=1024),
    dict(type=PeftSaveHook),
]
