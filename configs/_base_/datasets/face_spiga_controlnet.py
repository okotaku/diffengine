train_pipeline = [
    {
        "type": "torchvision/Resize",
        "size": 512,
        "interpolation": "bilinear",
        "keys": ["img", "condition_img"],
    },
    {
        "type": "RandomCrop",
        "size": 512,
        "keys": ["img", "condition_img"],
    },
    {
        "type": "RandomHorizontalFlip",
        "p": 0.5,
        "keys": ["img", "condition_img"],
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
        "input_keys": ["img", "condition_img", "text"],
    },
]
train_dataloader = {
    "batch_size": 4,
    "num_workers": 4,
    "dataset": {
        "type": "HFControlNetDataset",
        "dataset": "multimodalart/facesyntheticsspigacaptioned",
        "condition_column": "spiga_seg",
        "caption_column": "image_caption",
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
        ["a close up of a man with a mohawkcut and a purple shirt"] * 4,
        "condition_image": [
            'https://datasets-server.huggingface.co/assets/multimodalart/facesyntheticsspigacaptioned/--/multimodalart--facesyntheticsspigacaptioned/train/1/spiga_seg/image.jpg'  # noqa
        ] * 4,
    },
    {
        "type": "ControlNetSaveHook",
    },
]
