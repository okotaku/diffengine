_base_ = [
    "../../configs/_base_/models/stable_diffusion_xl_lora.py",
    "_base_/zunko_dreambooth_xl.py",
    "../../configs/_base_/schedules/stable_diffusion_500.py",
    "../../configs/_base_/default_runtime.py",
]

train_dataloader = {
    "dataset": {
        "class_image_config": {
            "model": {{_base_.model.model}},
        },
    },
}
