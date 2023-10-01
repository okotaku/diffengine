_base_ = [
    '../_base_/models/stable_diffusion_xl_ip_adapter_plus.py',
    '../_base_/datasets/pokemon_blip_xl_ip_adapter.py',
    '../_base_/schedules/stable_diffusion_xl_50e.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=1)

optim_wrapper_cfg = dict(accumulative_counts=4)  # update every four times

train_cfg = dict(by_epoch=True, max_epochs=100)
