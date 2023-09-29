train_pipeline = [
    dict(type='SaveImageShape'),
    dict(type='torchvision/Resize', size=1024, interpolation='bilinear'),
    dict(type='RandomCrop', size=1024),
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='ComputeTimeIds'),
    dict(type='torchvision/ToTensor'),
    dict(type='torchvision/Normalize', mean=[0.5], std=[0.5]),
    dict(
        type='PackInputs',
        input_keys=[
            'img', 'time_ids', 'prompt_embeds', 'pooled_prompt_embeds'
        ]),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='HFDatasetPreComputeEmbs',
        dataset='lambdalabs/pokemon-blip-captions',
        text_hasher='text_pokemon_blip',
        model='stabilityai/stable-diffusion-xl-base-1.0',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type='VisualizationHook',
        prompt=['yoda pokemon'] * 4,
        height=1024,
        width=1024),
    dict(type='SDCheckpointHook')
]
