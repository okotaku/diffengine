train_pipeline = [
    {
        "type": "SaveImageShape",
    },
    {
        "type": "CLIPImageProcessor",
    },
    {
        "type": "RandomTextDrop",
    },
    {
        "type": "torchvision/Resize",
        "size": 1024,
        "interpolation": "bilinear",
    },
    {
        "type": "RandomCrop",
        "size": 1024,
    },
    {
        "type": "RandomHorizontalFlip",
        "p": 0.5,
    },
    {
        "type": "ComputeTimeIds",
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
        "input_keys": ["img", "text", "time_ids", "clip_img"],
    },
]
train_dataloader = {
    "batch_size": 2,
    "num_workers": 2,
    "dataset": {
        "type": "HFDataset",
        "dataset": "lambdalabs/pokemon-blip-captions",
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
        "prompt":
        ["a drawing of a green pokemon with red eyes"] * 2 + [""] * 2,
        "example_image": [
            'https://datasets-server.huggingface.co/assets/lambdalabs/pokemon-blip-captions/--/default/train/0/image/image.jpg'  # noqa
        ] * 4,
        "height":
        1024,
        "width":
        1024,
    },
    {
        "type": "IPAdapterSaveHook",
    },
]
