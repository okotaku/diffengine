import torchvision
from mmengine.dataset import DefaultSampler

from diffengine.datasets import HFDataset
from diffengine.datasets.transforms import (
    ComputeTimeIds,
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
    RandomTextDrop,
    SaveImageShape,
    TorchVisonTransformWrapper,
    TransformersImageProcessor,
)
from diffengine.engine.hooks import IPAdapterSaveHook, VisualizationHook

train_pipeline = [
    dict(type=SaveImageShape),
    dict(type=TransformersImageProcessor,
         pretrained="google/siglip-so400m-patch14-384"),
    dict(type=RandomTextDrop),
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
    dict(
        type=PackInputs, input_keys=["img", "text", "time_ids", "clip_img"]),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
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
    dict(
        type=VisualizationHook,
        prompt=["a drawing of a green pokemon with red eyes"] * 2 + [""] * 2,
        example_image=[
            'https://github.com/LambdaLabsML/examples/blob/main/stable-diffusion-finetuning/README_files/README_2_0.png?raw=true'  # noqa
        ] * 4,
        height=1024,
        width=1024),
    dict(type=IPAdapterSaveHook),
]
