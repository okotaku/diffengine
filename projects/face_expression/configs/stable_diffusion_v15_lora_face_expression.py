from mmengine.config import read_base

with read_base():
    from diffengine.configs._base_.default_runtime import *
    from diffengine.configs._base_.models.stable_diffusion_v15_lora_textencoder import *  # noqa
    from diffengine.configs._base_.schedules.stable_diffusion_50e import *
    from ._base_.face_expression_dataset import *

model.update(
    dict(model='stablediffusionapi/anything-v5', lora_config=dict(rank=128)))

optim_wrapper.update(dict(optimizer=dict(lr=1e-4)))

train_cfg.update(dict(by_epoch=True, max_epochs=100))
