from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_amused_512 import *
    from .._base_.default_runtime import *
    from .._base_.models.amused_512 import *
    from .._base_.schedules.stable_diffusion_50e import *

optim_wrapper = dict(
    type=AmpOptimWrapper,
    dtype="float16",
    optimizer=dict(type=AdamW, lr=1e-4, weight_decay=1e-2),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            ".mlm_ln.weight": dict(decay_mult=0.0),
            ".embeddings.weight": dict(decay_mult=0.0),
        }))
