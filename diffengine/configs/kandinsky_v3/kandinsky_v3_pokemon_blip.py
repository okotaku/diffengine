from mmengine._strategy import ColossalAIStrategy
from mmengine.config import read_base
from mmengine.runner import FlexibleRunner

with read_base():
    from .._base_.datasets.pokemon_blip_kandinsky_v3 import *
    from .._base_.default_runtime import *
    from .._base_.models.kandinsky_v3 import *
    from .._base_.schedules.stable_diffusion_50e import *


optim_wrapper = dict(
    optimizer=dict(
        type="HybridAdam",
        lr=1e-5,
        weight_decay=1e-2),
    accumulative_counts=4)

default_hooks.update(
    checkpoint=dict(save_param_scheduler=False))  # no scheduler in this config

runner_type = FlexibleRunner
strategy = dict(type=ColossalAIStrategy,
                plugin=dict(type="LowLevelZeroPlugin",
                            stage=2,
                            precision="bf16",
                            max_norm=1.0))
