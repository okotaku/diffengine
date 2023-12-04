_base_ = [
    "../_base_/models/pixart_alpha_1024.py",
    "../_base_/datasets/pokemon_blip_pixart.py",
    "../_base_/schedules/stable_diffusion_50e.py",
    "../_base_/default_runtime.py",
]

optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(type="AdamW", lr=1e-5, weight_decay=3e-2),
    clip_grad=dict(max_norm=1.0))
