optim_wrapper = {
    "type": "AmpOptimWrapper",
    "dtype": "bfloat16",
    "optimizer": {
        "type": "Adafactor",
        "lr": 1e-5,
        "weight_decay": 1e-2,
        "scale_parameter": False,
        "relative_step": False,
    },
    "clip_grad": {
        "max_norm": 1.0,
    },
}

# train, val, test setting
train_cfg = {"by_epoch": True, "max_epochs": 50}
val_cfg = None
test_cfg = None

default_hooks = {
    "checkpoint": {
        "type": "CheckpointHook",
        "interval": 1,
        "max_keep_ckpts": 3,
    },
}
