_base_ = [
    '../_base_/models/stable_diffusion_xl_t2i_adapter.py',
    '../_base_/datasets/fill50k_t2i_adapter_xl.py',
    '../_base_/schedules/stable_diffusion_3e.py',
    '../_base_/default_runtime.py'
]

optim_wrapper = dict(
    optimizer=dict(lr=1e-5),
    accumulative_counts=2,
)
