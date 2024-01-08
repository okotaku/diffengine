import torchvision
from mmengine.dataset import InfiniteSampler

from diffengine.datasets import HFDreamBoothDataset
from diffengine.datasets.transforms import (
    ComputeTimeIds,
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
    SaveImageShape,
    TorchVisonTransformWrapper,
)
from diffengine.engine.hooks import PeftSaveHook, VisualizationHook

train_pipeline = [
    dict(type=SaveImageShape),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Resize,
         size=1024, interpolation="bilinear"),
    dict(type=RandomCrop, size=1024),
    dict(type=RandomHorizontalFlip, p=0.5),
    dict(type=ComputeTimeIds),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.ToTensor),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Normalize, mean=[0.5], std=[0.5]),
    dict(type=PackInputs, input_keys=["img", "text", "time_ids"]),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=HFDreamBoothDataset,
        dataset="diffusers/dog-example",
        instance_prompt="a photo of sks dog",
        pipeline=train_pipeline,
        class_prompt=None),
    sampler=dict(type=InfiniteSampler, shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type=VisualizationHook,
        prompt=["A photo of sks dog in a bucket"] * 4,
        by_epoch=False,
        interval=100,
        height=1024,
        width=1024),
    dict(type=PeftSaveHook),
]
