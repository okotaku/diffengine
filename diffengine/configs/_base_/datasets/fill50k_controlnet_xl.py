import torchvision
from mmengine.dataset import DefaultSampler

from diffengine.datasets import HFControlNetDataset
from diffengine.datasets.transforms import (
    ComputeTimeIds,
    DumpImage,
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
    SaveImageShape,
    TorchVisonTransformWrapper,
)
from diffengine.engine.hooks import ControlNetSaveHook, VisualizationHook

train_pipeline = [
    dict(type=SaveImageShape),
    dict(
        type=TorchVisonTransformWrapper,
        transform=torchvision.transforms.Resize,
        size=1024,
        interpolation="bilinear",
        keys=["img", "condition_img"]),
    dict(type=RandomCrop, size=1024, keys=["img", "condition_img"]),
    dict(type=RandomHorizontalFlip, p=0.5, keys=["img", "condition_img"]),
    dict(type=ComputeTimeIds),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.ToTensor, keys=["img", "condition_img"]),
    dict(type=DumpImage, max_imgs=10, dump_dir="work_dirs/dump"),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Normalize, mean=[0.5], std=[0.5]),
    dict(
        type=PackInputs,
        input_keys=["img", "condition_img", "text", "time_ids"]),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=HFControlNetDataset,
        dataset="fusing/fill50k",
        condition_column="conditioning_image",
        caption_column="text",
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
        prompt=["cyan circle with brown floral background"] * 4,
        condition_image=[
            'https://github.com/okotaku/diffengine/assets/24734142/1af9dbb0-b056-435c-bc4b-62a823889191'  # noqa
        ] * 4,
        height=1024,
        width=1024),
    dict(type=ControlNetSaveHook),
]
