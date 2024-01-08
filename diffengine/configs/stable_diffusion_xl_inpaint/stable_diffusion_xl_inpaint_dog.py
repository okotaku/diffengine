from mmengine.config import read_base
from mmengine.optim import AmpOptimWrapper
from transformers import Adafactor

with read_base():
    from .._base_.datasets.dog_inpaint_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl_inpaint import *
    from .._base_.schedules.stable_diffusion_1k import *



optim_wrapper = dict(
    type=AmpOptimWrapper,
    dtype="bfloat16",
    optimizer=dict(
        type=Adafactor,
        lr=1e-5,
        weight_decay=1e-2,
        scale_parameter=False,
        relative_step=False),
    clip_grad=dict(max_norm=1.0))
