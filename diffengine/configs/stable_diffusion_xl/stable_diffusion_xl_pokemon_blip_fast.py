from mmengine.config import read_base

from diffengine.engine.hooks import (
    CompileHook,
    FastNormHook,
    SDCheckpointHook,
    VisualizationHook,
)

with read_base():
    from .._base_.datasets.pokemon_blip_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl import *
    from .._base_.schedules.stable_diffusion_xl_50e import *


model.update(
    gradient_checkpointing=False)

train_dataloader.update(batch_size=1)

optim_wrapper.update(
    dtype="float16",
    accumulative_counts=4)

env_cfg.update(
    cudnn_benchmark=True,
)

custom_hooks = [
    dict(
        type=VisualizationHook,
        prompt=["yoda pokemon"] * 4,
        height=1024,
        width=1024),
    dict(type=SDCheckpointHook),
    dict(type=FastNormHook, fuse_main_ln=False, fuse_gn=False),
    dict(type=CompileHook, compile_main=True),
]
