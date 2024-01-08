from mmengine.config import read_base

from diffengine.engine.hooks import (
    LCMEMAUpdateHook,
    SDCheckpointHook,
    VisualizationHook,
)

with read_base():
    from .._base_.datasets.pokemon_blip_xl_pre_compute import *
    from .._base_.default_runtime import *
    from .._base_.models.lcm_xl import *
    from .._base_.schedules.lcm_xl_50e import *


train_dataloader.update(batch_size=2)

optim_wrapper.update(accumulative_counts=2)  # update every four times

custom_hooks = [
    dict(
        type=VisualizationHook,
        prompt=["yoda pokemon"] * 4,
        height=1024,
        width=1024),
    dict(type=SDCheckpointHook),
    dict(type=LCMEMAUpdateHook),
]
