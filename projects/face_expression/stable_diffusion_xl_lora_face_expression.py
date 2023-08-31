_base_ = [
    '../../configs/_base_/models/stable_diffusion_xl_lora.py',
    '_base_/face_expression_xl_dataset.py',
    '../../configs/_base_/schedules/stable_diffusion_50e.py',
    '../../configs/_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=1)

optim_wrapper = dict(optimizer=dict(lr=1e-4))

model = dict(
    model='Linaqruf/animagine-xl', vae_model=None, lora_config=dict(rank=128))

train_cfg = dict(by_epoch=True, max_epochs=100)
