_base_ = [
    "../_base_/models/stable_diffusion_xl_controlnetxs.py",
    "../_base_/datasets/fill50k_controlnet_xl.py",
    "../_base_/schedules/stable_diffusion_3e.py",
    "../_base_/default_runtime.py",
]

train_dataloader = dict(batch_size=4)

optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
    accumulative_counts=2,
)
