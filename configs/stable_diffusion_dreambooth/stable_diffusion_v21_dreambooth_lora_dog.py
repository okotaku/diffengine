_base_ = [
    "../_base_/models/stable_diffusion_v21_lora.py",
    "../_base_/datasets/dog_dreambooth.py",
    "../_base_/schedules/stable_diffusion_1k.py",
    "../_base_/default_runtime.py",
]

train_pipeline = [
    {
        "type": "torchvision/Resize",
        "size": 768,
        "interpolation": "bilinear",
    },
    {
        "type": "RandomCrop",
        "size": 768,
    },
    {
        "type": "RandomHorizontalFlip",
        "p": 0.5,
    },
    {
        "type": "torchvision/ToTensor",
    },
    {
        "type": "torchvision/Normalize",
        "mean": [0.5],
        "std": [0.5],
    },
    {
        "type": "PackInputs",
    },
]
train_dataloader = {
    "dataset": {
        "class_image_config": {
            "model": {{_base_.model.model}},
        },
        "pipeline": train_pipeline,
    },
}
