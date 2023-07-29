_base_ = [
    '../_base_/models/stable_diffusion_v15.py',
    '../_base_/datasets/pokemon_blip.py',
    '../_base_/schedules/stable_diffusion_v15_50e.py',
    '../_base_/default_runtime.py'
]

model = dict(loss=dict(type='SNRL2Loss', snr_gamma=5.0, loss_weight=1.0))
