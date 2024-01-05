_base_ = [
    "../_base_/models/deepfloyd_if_xl_lora.py",
    "../_base_/datasets/pokemon_blip_if.py",
    "../_base_/schedules/stable_diffusion_50e.py",
    "../_base_/default_runtime.py",
]

optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(type="AdamW", lr=5e-6, weight_decay=1e-2),
    clip_grad=dict(max_norm=1.0))
