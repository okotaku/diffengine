from mmengine.config import read_base

with read_base():
    from diffengine.configs._base_.default_runtime import *
    from diffengine.configs._base_.schedules.stable_diffusion_50e import *

    from ._base_.counterfeit_xl_lora import *
    from ._base_.face_expression_xl_dataset import *


train_dataloader.update(batch_size=2)

optim_wrapper.update(optimizer=dict(lr=1e-4), accumulative_counts=2)

train_cfg.update(by_epoch=True, max_epochs=50)
