from mmengine.config import read_base

with read_base():
    from diffengine.configs._base_.datasets.fill50k_controlnet_xl import *
    from diffengine.configs._base_.default_runtime import *
    from diffengine.configs._base_.schedules.stable_diffusion_3e import *

    from ._base_.stable_diffusion_xl_controlnetxs import *


train_dataloader.update(batch_size=4)

optim_wrapper.update(
    optimizer=dict(lr=1e-4),
    accumulative_counts=2,
)
