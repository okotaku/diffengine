from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl import *
    from .._base_.schedules.stable_diffusion_xl_50e import *


model.update(timesteps_generator=dict(type="RangeTimeSteps"))

train_dataloader.update(batch_size=1)

optim_wrapper.update(accumulative_counts=4)  # update every four times
