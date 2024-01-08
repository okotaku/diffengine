from mmengine.config import read_base

with read_base():
    from diffengine.configs._base_.default_runtime import *
    from diffengine.configs._base_.schedules.stable_diffusion_50e import *

    from ._base_.anythingv5_lora_textencoder import *
    from ._base_.face_expression_dataset import *


optim_wrapper.update(optimizer=dict(lr=1e-4))

train_cfg.update(by_epoch=True, max_epochs=100)
