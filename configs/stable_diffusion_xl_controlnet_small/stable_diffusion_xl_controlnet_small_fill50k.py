_base_ = [
    "../_base_/models/stable_diffusion_xl_controlnet.py",
    "../_base_/datasets/fill50k_controlnet_xl.py",
    "../_base_/schedules/stable_diffusion_3e.py",
    "../_base_/default_runtime.py",
]

model = {"transformer_layers_per_block": [0, 0, 0]}

train_dataloader = {"batch_size": 4}

optim_wrapper = {
    "optimizer": {
        "lr": 3e-5,
    },
    "accumulative_counts": 2,
}
