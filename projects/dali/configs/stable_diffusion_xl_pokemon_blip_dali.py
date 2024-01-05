_base_ = [
    "../../../configs/_base_/models/stable_diffusion_xl.py",
    "../../../configs/_base_/datasets/pokemon_blip_xl.py",
    "../../../configs/_base_/schedules/stable_diffusion_xl_50e.py",
    "../../../configs/_base_/default_runtime.py",
]

custom_imports = dict(imports=["projects.dali"], allow_failed_imports=False)

model = dict(
    gradient_checkpointing=False)

train_dataloader = dict(batch_size=1)

#optim_wrapper = dict(
#    dtype="bfloat16",
#    accumulative_counts=4)

optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(
        type="Adafactor",
        lr=1e-5,
        weight_decay=1e-2,
        scale_parameter=False,
        relative_step=False),
    clip_grad=dict(max_norm=1.0),
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
    dict(type="FastNormHook", fuse_main_ln=False, fuse_gn=False),
    dict(type="CompileHook", compile_main=True),
]
