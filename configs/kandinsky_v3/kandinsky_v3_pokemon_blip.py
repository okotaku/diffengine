_base_ = [
    "../_base_/models/kandinsky_v3.py",
    "../_base_/datasets/pokemon_blip_kandinsky_v3.py",
    "../_base_/schedules/stable_diffusion_xl_50e.py",
    "../_base_/default_runtime.py",
]

optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(
        type="HybridAdam",
        lr=1e-5,
        weight_decay=1e-2),
    accumulative_counts=4)

default_hooks = dict(
    checkpoint=dict(save_param_scheduler=False))  # no scheduler in this config

runner_type = "FlexibleRunner"
strategy = dict(type="ColossalAIStrategy",
                plugin=dict(type="LowLevelZeroPlugin",
                            stage=2,
                            precision="bf16",
                            max_norm=1.0))
