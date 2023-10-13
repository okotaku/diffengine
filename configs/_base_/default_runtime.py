default_scope = "diffengine"

env_cfg = {
    "cudnn_benchmark": False,
    "mp_cfg": {
        "mp_start_method": "fork",
        "opencv_num_threads": 4,
    },
    "dist_cfg": {
        "backend": "nccl",
    },
}

load_from = None
resume = False
randomness = {"seed": None, "deterministic": False}
