_base_ = [
    '../_base_/models/stable_diffusion_xl_esd.py',
    '../_base_/datasets/gogh_esd_xl.py',
    '../_base_/schedules/stable_diffusion_500.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=1)

optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(
        type='Adafactor',
        lr=1e-5,
        weight_decay=1e-2,
        scale_parameter=False,
        relative_step=False),
    clip_grad=dict(max_norm=1.0))

train_cfg = dict(max_iters=300)
