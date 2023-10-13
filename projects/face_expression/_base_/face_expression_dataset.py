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
        "type": "torchvision/Normalize",
        "mean": [0.5],
        "std": [0.5],
    },
    {
        "type": "PackInputs",
    },
]
train_dataloader = {
    "batch_size": 1,
    "num_workers": 4,
    "dataset": {
        "type": "HFDataset",
        "dataset": "data/ExpressionTraining",
        "pipeline": train_pipeline,
        "image_column": "file_name",
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
        "prompt": [
            "1girl, >_<, blue hair",
            "1girl, X X, blue hair",
            "1girl, @_@, blue hair",
            "1girl, =_=, blue hair",
        ],
    },
    {
        "type": "LoRASaveHook",
    },
]
