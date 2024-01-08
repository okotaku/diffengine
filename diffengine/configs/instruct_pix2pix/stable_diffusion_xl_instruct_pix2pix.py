from mmengine.config import read_base

with read_base():
    from .._base_.datasets.instructpix2pix_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl_instruct_pix2pix import *
    from .._base_.schedules.stable_diffusion_3e import *


optim_wrapper.update(
    optimizer=dict(
        type="Adafactor",
        lr=3e-5,
        weight_decay=1e-2,
        scale_parameter=False,
        relative_step=False),
    accumulative_counts=4)
