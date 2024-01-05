_base_ = [
    "../_base_/models/stable_diffusion_xl.py",
    "../_base_/datasets/pokemon_blip_xl.py",
    "../_base_/schedules/stable_diffusion_xl_50e.py",
    "../_base_/default_runtime.py",
]

model = dict(
    gradient_checkpointing=False)

train_dataloader = dict(batch_size=8, num_workers=8)

optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(
        type="HybridAdam",
        lr=1e-5,
        weight_decay=1e-2),
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

default_hooks = dict(
    checkpoint=dict(save_param_scheduler=False))  # no scheduler in this config

runner_type = "FlexibleRunner"
strategy = dict(type="ColossalAIStrategy",
                mixed_precision="fp16",
                plugin=dict(type="LowLevelZeroPlugin",
                            stage=2,
                            max_norm=1.0))
