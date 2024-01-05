_base_ = [
    "../_base_/models/stable_diffusion_xl_inpaint.py",
    "../_base_/datasets/dog_inpaint_xl.py",
    "../_base_/schedules/stable_diffusion_1k.py",
    "../_base_/default_runtime.py",
]

optim_wrapper = dict(
    _delete_=True,
    type="AmpOptimWrapper",
    dtype="bfloat16",
    optimizer=dict(
        type="Adafactor",
        lr=1e-5,
        weight_decay=1e-2,
        scale_parameter=False,
        relative_step=False),
    clip_grad=dict(max_norm=1.0))
