_base_ = [
    '../../configs/_base_/models/stable_diffusion_xl_lora.py',
    '_base_/zunko_dreambooth_xl.py',
    '../../configs/_base_/schedules/stable_diffusion_500.py',
    '../../configs/_base_/default_runtime.py'
]

model = dict(model='gsdf/CounterfeitXL')

train_dataloader = dict(
    dataset=dict(
        class_image_config=dict(model={{_base_.model.model}}),
        instance_prompt='1girl, sks'))

custom_hooks = [
    dict(
        type='VisualizationHook',
        prompt=['1girl, sks, in a bucket'] * 4,
        by_epoch=False,
        interval=100),
    dict(type='LoRASaveHook'),
]
