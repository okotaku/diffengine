_base_ = [
    "../_base_/models/stable_diffusion_xl_esd.py",
    "../_base_/datasets/gogh_esd_xl.py",
    "../_base_/schedules/stable_diffusion_500.py",
    "../_base_/default_runtime.py",
]

train_dataloader = {"batch_size": 1}

optim_wrapper = {
    "_delete_": True,
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

train_cfg = {"max_iters": 300}
