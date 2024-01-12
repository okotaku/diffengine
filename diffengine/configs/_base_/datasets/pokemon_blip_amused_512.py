import torchvision
from mmengine.dataset import DefaultSampler

from diffengine.datasets import HFDataset
from diffengine.datasets.transforms import (
    ComputeaMUSEdMicroConds,
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
    RandomTextDrop,
    SaveImageShape,
    TorchVisonTransformWrapper,
)
from diffengine.engine.hooks import TransformerCheckpointHook, VisualizationHook

train_pipeline = [
    dict(type=SaveImageShape),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Resize,
         size=512, interpolation="bilinear"),
    dict(type=RandomCrop, size=512),
    dict(type=RandomHorizontalFlip, p=0.5),
    dict(type=ComputeaMUSEdMicroConds),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.ToTensor),
    dict(type=RandomTextDrop),
    dict(type=PackInputs, input_keys=["img", "text", "micro_conds"]),
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=HFDataset,
        dataset="lambdalabs/pokemon-blip-captions",
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(type=VisualizationHook, prompt=["yoda pokemon"] * 4),
    dict(type=TransformerCheckpointHook),
]
