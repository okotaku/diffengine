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
        "type":
        "PackInputs",
        "input_keys": [
            "img",
            "time_ids",
            "prompt_embeds",
            "pooled_prompt_embeds",
        ],
    },
]
train_dataloader = {
    "batch_size": 2,
    "num_workers": 2,
    "dataset": {
        "type": "HFDatasetPreComputeEmbs",
        "dataset": "lambdalabs/pokemon-blip-captions",
        "text_hasher": "text_pokemon_blip",
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
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
        "type": "VisualizationHook",
        "prompt": ["yoda pokemon"] * 4,
        "height": 1024,
        "width": 1024,
    },
    {
        "type": "SDCheckpointHook",
    },
]
