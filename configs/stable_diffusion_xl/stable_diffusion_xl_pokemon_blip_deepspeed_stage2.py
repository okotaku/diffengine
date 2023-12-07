_base_ = [
    "../_base_/models/stable_diffusion_xl.py",
    "../_base_/datasets/pokemon_blip_xl.py",
    "../_base_/schedules/stable_diffusion_xl_50e.py",
    "../_base_/default_runtime.py",
]

model = dict(
    enable_xformers=True,
    gradient_checkpointing=False)

train_dataloader = dict(batch_size=8, num_workers=8)

optim_wrapper = dict(
    _delete_=True,
    type="DeepSpeedOptimWrapper",
    optimizer=dict(
        type="FusedAdam",
        lr=1e-5,
        weight_decay=1e-2))

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
]

runner_type = "FlexibleRunner"
strategy = dict(
    type="DeepSpeedStrategy",
    gradient_clipping=1.0,
    gradient_accumulation_steps=4,
    fp16=dict(
        enabled=True,
        fp16_master_weights_and_grads=False,
        loss_scale=0,
        loss_scale_window=500,
        hysteresis=2,
        min_loss_scale=1,
        initial_scale_power=15,
    ),
    inputs_to_half=["inputs"],
    zero_optimization=dict(
        stage=2,
        allgather_partitions=True,
        reduce_scatter=True,
        allgather_bucket_size=50000000,
        reduce_bucket_size=50000000,
        overlap_comm=False,
        contiguous_gradients=True,
        cpu_offload=False),
)
