from mmengine.config import read_base

with read_base():
    from .._base_.datasets.face_spiga_controlnet import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15_controlnet import *
    from .._base_.schedules.stable_diffusion_3e import *
