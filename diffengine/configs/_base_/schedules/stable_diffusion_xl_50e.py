from mmengine.hooks import CheckpointHook
from mmengine.optim import AmpOptimWrapper
from transformers.optimization import Adafactor

optim_wrapper = dict(
    type=AmpOptimWrapper,
    dtype="bfloat16",
    optimizer=dict(
        type=Adafactor,
        lr=1e-5,
        weight_decay=1e-2,
        scale_parameter=False,
        relative_step=False),
    clip_grad=dict(max_norm=1.0))

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=50)
val_cfg = None
test_cfg = None

default_hooks = dict(
    checkpoint=dict(
        type=CheckpointHook,
        interval=1,
        max_keep_ckpts=3,
    ))
