_base_ = [
    "../../configs/_base_/models/stable_diffusion_v15_lora.py",
    "_base_/zunko_dreambooth.py",
    "../../configs/_base_/schedules/stable_diffusion_1k.py",
    "../../configs/_base_/default_runtime.py",
]

train_dataloader = dict(
    dataset=dict(class_image_config=dict(model={{_base_.model.model}})))
