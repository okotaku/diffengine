train_pipeline = [
    {
        "type": "SaveImageShape",
    },
    {
        "type": "torchvision/Resize",
        "size": 1024,
        "interpolation": "bilinear",
        "keys": ["img", "condition_img"],
    },
    {
        "type": "RandomCrop",
        "size": 1024,
        "keys": ["img", "condition_img"],
    },
    {
        "type": "RandomHorizontalFlip",
        "p": 0.5,
        "keys": ["img", "condition_img"],
    },
    {
        "type": "ComputeTimeIds",
    },
    {
        "type": "torchvision/ToTensor",
        "keys": ["img", "condition_img"],
    },
    {
        "type": "DumpImage",
        "max_imgs": 10,
        "dump_dir": "work_dirs/dump",
    },
    {
        "type": "torchvision/Normalize",
        "mean": [0.5],
        "std": [0.5],
    },
    {
        "type": "PackInputs",
        "input_keys": ["img", "condition_img", "text", "time_ids"],
    },
]
train_dataloader = {
    "batch_size": 2,
    "num_workers": 4,
    "dataset": {
        "type": "HFControlNetDataset",
        "dataset": "fusing/fill50k",
        "condition_column": "conditioning_image",
        "caption_column": "text",
        "pipeline": train_pipeline,
    },
    "sampler": {
        "type": "DefaultSampler",
        "shuffle": True,
    },
}

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    {
        "type":
        "VisualizationHook",
        "prompt": ["cyan circle with brown floral background"] * 4,
        "condition_image": [
            'https://datasets-server.huggingface.co/assets/fusing/fill50k/--/default/train/74/conditioning_image/image.jpg'  # noqa
        ] * 4,
        "height":
        1024,
        "width":
        1024,
    },
    {
        "type": "ControlNetSaveHook",
    },
]
