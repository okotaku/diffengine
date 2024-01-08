from mmengine.config import read_base

with read_base():
    from .._base_.datasets.gogh_esd_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl_esd import *
    from .._base_.schedules.stable_diffusion_500 import *


train_dataloader.update(batch_size=1)

optim_wrapper = dict(
    optimizer=dict(
        type="Adafactor",
        lr=1e-5,
        weight_decay=1e-2,
        scale_parameter=False,
        relative_step=False),
    clip_grad=dict(max_norm=1.0))

train_cfg.update(max_iters=300)
