_base_ = [
    '../_base_/models/stable_diffusion_xl_lora.py',
    '../_base_/datasets/pokemon_blip_xl.py',
    '../_base_/schedules/stable_diffusion_v15_50e.py',
    '../_base_/default_runtime.py'
]

custom_hooks = [
    dict(type='VisualizationHook', prompt=['yoda pokemon'] * 4),
    dict(type='LoRASaveHook'),
    dict(type='UnetEMAHook', momentum=1e-4, priority='ABOVE_NORMAL')
]
