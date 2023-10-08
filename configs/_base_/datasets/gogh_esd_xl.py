train_pipeline = [
    dict(
        type='PackInputs',
        input_keys=[
            'text', 'prompt_embeds', 'pooled_prompt_embeds',
            'null_prompt_embeds', 'null_pooled_prompt_embeds'
        ]),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='HFESDDatasetPreComputeEmbs',
        forget_caption='Van Gogh',
        model='stabilityai/stable-diffusion-xl-base-1.0',
        pipeline=train_pipeline),
    sampler=dict(type='InfiniteSampler', shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type='VisualizationHook',
        prompt=['The starry night by Van Gogh'] * 4,
        by_epoch=False,
        interval=100,
        height=1024,
        width=1024),
    dict(type='SDCheckpointHook')
]
