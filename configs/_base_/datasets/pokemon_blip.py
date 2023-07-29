train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(type='HFDataset', dataset='lambdalabs/pokemon-blip-captions'),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(type='VisualizationHook', prompt=['yoda pokemon'] * 4),
    dict(type='SDCheckpointHook')
]
