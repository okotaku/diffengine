optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=1e-2),
    clip_grad=dict(max_norm=1.0))

# train, val, test setting
train_cfg = dict(type='IterBasedTrainLoop', max_iters=2000)
val_cfg = None
test_cfg = None

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=100,
        by_epoch=False,
        max_keep_ckpts=3,
    ), )
log_processor = dict(by_epoch=False)
