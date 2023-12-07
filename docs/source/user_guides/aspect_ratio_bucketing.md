# Aspect Ratio Bucketing

Training with aspect ratio bucketing can greatly improve the quality of outputs.
For more details, you can check [NovelAI Aspect Ratio Bucketing](https://github.com/NovelAI/novelai-aspect-ratio-bucketing).

## Finetune

To use Aspect Ratio Bucketing in finetune, you need to follow these steps:

1. Fix the dataset config.

Change `torchvision/Resize` and `RandomCrop` to `MultiAspectRatioResizeCenterCrop`. Also, use `AspectRatioBatchSampler`.

```
train_pipeline = [
    dict(type="SaveImageShape"),
    dict(type='MultiAspectRatioResizeCenterCrop',
         sizes=[
             [640, 1536], [768, 1344], [832, 1216], [896, 1152],
             [1024, 1024], [1152, 896], [1216, 832], [1344, 768], [1536, 640]
             ],
         interpolation='bilinear'),
    dict(type="RandomHorizontalFlip", p=0.5),
    dict(type="ComputeTimeIds"),
    dict(type="torchvision/ToTensor"),
    dict(type="torchvision/Normalize", mean=[0.5], std=[0.5]),
    dict(type="PackInputs", input_keys=["img", "text", "time_ids"]),
]
train_dataloader = dict(
    ...
    dataset=dict(
        ...
        pipeline=train_pipeline),
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
)
```

2. Run training.

## ControlNet

To use Aspect Ratio Bucketing in ControlNet, you need to follow these steps:

1. Fix dataset config.

```
train_pipeline = [
    dict(type="SaveImageShape"),
    dict(
        type="MultiAspectRatioResizeCenterCrop",
        sizes=[
             [640, 1536], [768, 1344], [832, 1216], [896, 1152],
             [1024, 1024], [1152, 896], [1216, 832], [1344, 768], [1536, 640]
             ],
        interpolation='bilinear',
        keys=["img", "condition_img"]),
    dict(type="RandomHorizontalFlip", p=0.5, keys=["img", "condition_img"]),
    dict(type="ComputeTimeIds"),
    dict(type="torchvision/ToTensor", keys=["img", "condition_img"]),
    dict(type="DumpImage", max_imgs=10, dump_dir="work_dirs/dump"),
    dict(type="torchvision/Normalize", mean=[0.5], std=[0.5]),
    dict(
        type="PackInputs",
        input_keys=["img", "condition_img", "text", "time_ids"]),
]
train_dataloader = dict(
    ...
    dataset=dict(
        ...
        pipeline=train_pipeline),
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
)
```

2. Run training.
