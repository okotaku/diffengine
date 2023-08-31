_base_ = [
    '../../configs/_base_/models/stable_diffusion_v15_lora_textencoder.py',
    '_base_/face_expression_dataset.py',
    '../../configs/_base_/schedules/stable_diffusion_50e.py',
    '../../configs/_base_/default_runtime.py'
]

model = dict(
    model='stablediffusionapi/anything-v5', lora_config=dict(rank=128))

optim_wrapper = dict(optimizer=dict(lr=1e-4))

train_cfg = dict(by_epoch=True, max_epochs=100)
