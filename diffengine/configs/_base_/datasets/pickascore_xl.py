import torchvision
from mmengine.dataset import DefaultSampler

from diffengine.datasets import HFDPODataset
from diffengine.datasets.transforms import (
    ComputeTimeIds,
    ConcatMultipleImgs,
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
    SaveImageShape,
    TorchVisonTransformWrapper,
)
from diffengine.engine.hooks import SDCheckpointHook, VisualizationHook

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
    dict(type=ConcatMultipleImgs),
    dict(type=PackInputs, input_keys=["img", "text", "time_ids"]),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=HFDPODataset,
        dataset="kashif/pickascore",
        split="validation",
        image_columns=["jpg_0", "jpg_1"],
        caption_column="caption",
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
        prompt=[
            "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",  # noqa
            "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
            "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
            "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",  # noqa
        ],
        height=1024,
        width=1024),
    dict(type=SDCheckpointHook),
]
