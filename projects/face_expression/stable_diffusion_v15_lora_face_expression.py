_base_ = [
    "../../configs/_base_/models/stable_diffusion_v15_lora_textencoder.py",
    "_base_/face_expression_dataset.py",
    "../../configs/_base_/schedules/stable_diffusion_50e.py",
    "../../configs/_base_/default_runtime.py",
]

model = {
    "model": "stablediffusionapi/anything-v5",
    "lora_config": {
        "rank": 128,
    },
}

optim_wrapper = {"optimizer": {"lr": 1e-4}}

train_cfg = {"by_epoch": True, "max_epochs": 100}
