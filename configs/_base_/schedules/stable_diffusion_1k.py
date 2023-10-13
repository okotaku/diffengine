optim_wrapper = {
    "type": "AmpOptimWrapper",
    "dtype": "float16",
    "optimizer": {
        "type": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-2,
    },
    "clip_grad": {
        "max_norm": 1.0,
    },
}

# train, val, test setting
train_cfg = {"type": "IterBasedTrainLoop", "max_iters": 1000}
val_cfg = None
test_cfg = None

default_hooks = {
    "checkpoint": {
        "type": "CheckpointHook",
        "interval": 100,
        "by_epoch": False,
        "max_keep_ckpts": 3,
    },
}
log_processor = {"by_epoch": False}
