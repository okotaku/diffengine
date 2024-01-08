from mmengine.config import read_base

with read_base():
    from .._base_.datasets.waterpainting_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl_lora import *
    from .._base_.schedules.stable_diffusion_500 import *

train_cfg.update(max_iters=2000)
