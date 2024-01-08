from mmengine.config import read_base

from diffengine.engine.hooks import PeftSaveHook, VisualizationHook

with read_base():
    from diffengine.configs._base_.default_runtime import *
    from diffengine.configs._base_.schedules.stable_diffusion_500 import *

    from ._base_.counterfeit_xl_lora import *
    from ._base_.zunko_dreambooth_xl import *


custom_hooks = [
    dict(
        type=VisualizationHook,
        prompt=["1girl, sks, in a bucket"] * 4,
        by_epoch=False,
        interval=100),
    dict(type=PeftSaveHook),
]
