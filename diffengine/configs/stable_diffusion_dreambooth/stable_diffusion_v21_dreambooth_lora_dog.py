from mmengine.config import read_base

with read_base():
    from .._base_.datasets.dog_dreambooth import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v21_lora import *
    from .._base_.schedules.stable_diffusion_1k import *


train_pipeline = [
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Resize,
         size=768, interpolation="bilinear"),
    dict(type=RandomCrop, size=768),
    dict(type=RandomHorizontalFlip, p=0.5),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.ToTensor),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Normalize, mean=[0.5], std=[0.5]),
    dict(type=PackInputs),
]
train_dataloader.update(
    dataset=dict(
        class_image_config=dict(model=model.model),
        pipeline=train_pipeline))
