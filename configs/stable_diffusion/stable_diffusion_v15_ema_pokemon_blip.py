_base_ = [
    '../_base_/models/stable_diffusion_v15.py',
    '../_base_/datasets/pokemon_blip.py',
    '../_base_/schedules/stable_diffusion_50e.py',
    '../_base_/default_runtime.py'
]

custom_hooks = [
    dict(type='VisualizationHook', prompt=['yoda pokemon'] * 4),
    dict(type='SDCheckpointHook'),
    dict(type='UnetEMAHook', momentum=1e-4, priority='ABOVE_NORMAL')
]
