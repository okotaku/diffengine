train_pipeline = [
    {
        "type": "SaveImageShape",
    },
    {
        "type": "torchvision/Resize",
        "size": 1024,
        "interpolation": "bilinear",
    },
    {
        "type": "CenterCrop",
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
        "input_keys": ["img", "text", "time_ids"],
    },
]
train_dataloader = {
    "batch_size": 2,
    "num_workers": 2,
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
        "height":
        1024,
        "width":
        1024,
    },
    {
        "type": "LoRASaveHook",
    },
]
