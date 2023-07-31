_base_ = [
    '../_base_/models/stable_diffusion_v15_lora_textencoder.py',
    '../_base_/datasets/dog_dreambooth.py',
    '../_base_/schedules/stable_diffusion_v15_2k.py',
    '../_base_/default_runtime.py'
]

custom_hooks = [
    dict(
        type='VisualizationHook',
        prompt=['A photo of sks dog in a bucket'] * 4,
        by_epoch=False,
        interval=100),
    dict(type='LoRASaveHook'),
    dict(type='UnetEMAHook', momentum=1e-4, priority='ABOVE_NORMAL')
]
