_base_ = [
    '../_base_/models/stable_diffusion_v21_lora.py',
    '../_base_/datasets/dog_dreambooth.py',
    '../_base_/schedules/stable_diffusion_v15_1k.py',
    '../_base_/default_runtime.py'
]

train_pipeline = [
    dict(type='torchvision/Resize', size=768, interpolation='bilinear'),
    dict(type='torchvision/RandomCrop', size=768),
    dict(type='torchvision/RandomHorizontalFlip', p=0.5),
    dict(type='torchvision/ToTensor'),
    dict(type='torchvision/Normalize', mean=[0.5], std=[0.5]),
    dict(type='PackInputs'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline), )
