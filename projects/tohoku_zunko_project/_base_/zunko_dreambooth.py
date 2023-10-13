train_pipeline = [
    {
        "type": "torchvision/Resize",
        "size": 512,
        "interpolation": "bilinear",
    },
    {
        "type": "RandomCrop",
        "size": 512,
    },
    {
        "type": "RandomHorizontalFlip",
        "p": 0.5,
    },
    {
        "type": "torchvision/ToTensor",
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
    },
]
train_dataloader = {
    "batch_size": 4,
    "num_workers": 4,
    "dataset": {
        "type": "HFDreamBoothDataset",
        "dataset": "data/zunko",
        "instance_prompt": "a photo of sks character",
        "pipeline": train_pipeline,
    },
    "sampler": {
        "type": "InfiniteSampler",
        "shuffle": True,
    },
}

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    {
        "type": "VisualizationHook",
        "prompt": ["A photo of sks character in a bucket"] * 4,
        "by_epoch": False,
        "interval": 100,
    },
    {
        "type": "LoRASaveHook",
    },
]
