from mmengine.config import read_base

from diffengine.engine.hooks import PeftSaveHook, VisualizationHook

with read_base():
    from .._base_.datasets.pokemon_blip_wuerstchen import *
    from .._base_.default_runtime import *
    from .._base_.models.wuerstchen_prior_lora import *
    from .._base_.schedules.stable_diffusion_50e import *


train_dataloader.update(
    batch_size=8,
)

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-2),
    clip_grad=dict(max_norm=1.0))

custom_hooks = [
    dict(type=VisualizationHook, prompt=["A robot pokemon, 4k photo"] * 4,
         height=768, width=768),
    dict(type=PeftSaveHook),
]
