_base_ = [
    "../_base_/models/stable_diffusion_xl_lora.py",
    "../_base_/datasets/waterpainting_xl.py",
    "../_base_/schedules/stable_diffusion_500.py",
    "../_base_/default_runtime.py",
]

train_cfg = dict(max_iters=2000)
