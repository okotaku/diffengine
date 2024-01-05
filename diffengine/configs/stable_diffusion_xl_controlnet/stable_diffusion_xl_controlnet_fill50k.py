_base_ = [
    "../_base_/models/stable_diffusion_xl_controlnet.py",
    "../_base_/datasets/fill50k_controlnet_xl.py",
    "../_base_/schedules/stable_diffusion_1e.py",
    "../_base_/default_runtime.py",
]

optim_wrapper = dict(
    optimizer=dict(lr=1e-5),
    accumulative_counts=2,
)
