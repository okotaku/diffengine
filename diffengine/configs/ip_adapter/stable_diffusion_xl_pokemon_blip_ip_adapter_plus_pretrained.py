from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_xl_ip_adapter import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl_ip_adapter_plus import *
    from .._base_.schedules.stable_diffusion_xl_50e import *


model.update(image_encoder=dict(subfolder="models/image_encoder"),
             pretrained_adapter="h94/IP-Adapter",
             pretrained_adapter_subfolder="sdxl_models",
             pretrained_adapter_weights_name="ip-adapter-plus_sdxl_vit-h.bin")

train_dataloader.update(batch_size=1)

optim_wrapper.update(accumulative_counts=4)  # update every four times

train_cfg.update(by_epoch=True, max_epochs=10)
