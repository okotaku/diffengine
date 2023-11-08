_base_ = [
    "../_base_/models/wuerstchen_prior.py",
    "../_base_/datasets/pokemon_blip_wuerstchen.py",
    "../_base_/schedules/stable_diffusion_50e.py",
    "../_base_/default_runtime.py",
]

optim_wrapper_cfg = dict(
    optimizer=dict(lr=1e-5),
    accumulative_counts=4)  # update every four times
