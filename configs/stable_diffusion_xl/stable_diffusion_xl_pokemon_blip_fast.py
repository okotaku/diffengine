_base_ = [
    "../_base_/models/stable_diffusion_xl.py",
    "../_base_/datasets/pokemon_blip_xl.py",
    "../_base_/schedules/stable_diffusion_xl_50e.py",
    "../_base_/default_runtime.py",
]

model = dict(
    gradient_checkpointing=False)

train_dataloader = dict(batch_size=1)

optim_wrapper = dict(
    dtype="float16",
    accumulative_counts=4)

env_cfg = dict(
    cudnn_benchmark=True,
)

custom_hooks = [
    dict(
        type="VisualizationHook",
        prompt=["yoda pokemon"] * 4,
        height=1024,
        width=1024),
    dict(type="SDCheckpointHook"),
    dict(type="FastNormHook", fuse_unet_ln=False, fuse_gn=False),
    dict(type="CompileHook", compile_unet=True),
]
