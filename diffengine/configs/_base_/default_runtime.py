default_scope = "diffengine"

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=4),
    dist_cfg=dict(backend="nccl"),
)

load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
