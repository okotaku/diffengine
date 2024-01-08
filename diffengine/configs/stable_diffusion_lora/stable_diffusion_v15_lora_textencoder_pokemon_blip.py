from mmengine.config import read_base

from diffengine.engine.hooks import PeftSaveHook, VisualizationHook

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15_lora_textencoder import *
    from .._base_.schedules.stable_diffusion_50e import *


model.update(
    unet_lora_config=dict(r=32,
        lora_alpha=32),
    text_encoder_lora_config=dict(r=32,
        lora_alpha=32))

custom_hooks = [
    dict(type=VisualizationHook, prompt=["yoda pokemon"] * 4),
    dict(type=PeftSaveHook),
]
