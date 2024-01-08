import argparse
import os.path as osp
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from diffengine.configs import cfgs_name_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a checkpoint to be published")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("in_file", help="input checkpoint filename")
    parser.add_argument("out_dir", help="output dir")
    parser.add_argument(
        "--save-keys",
        nargs="+",
        type=str,
        default=["unet", "text_encoder", "transformer"],
        help="keys to save in the published checkpoint")
    parser.add_argument(
        "--colossalai",
        action="store_true",
        help="whether the checkpoint is trained with colossalai")
    return parser.parse_args()


def process_checkpoint(runner, out_dir, save_keys=None) -> None:
    if save_keys is None:
        save_keys = ["unet", "text_encoder"]
    for k in save_keys:
        model = getattr(runner.model, k)
        model.save_pretrained(osp.join(out_dir, k))
    print_log(f"The published model is saved at {out_dir}.", logger="current")


def main() -> None:
    args = parse_args()
    allowed_save_keys = ["unet", "text_encoder", "transformer"]
    if not set(args.save_keys).issubset(set(allowed_save_keys)):
        msg = (f"These metrics are supported: {allowed_save_keys}, "
               f"but got {args.save_keys}")
        raise KeyError(msg)

    # parse config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError as exc:
            msg = f"Cannot find {args.config}"
            raise FileNotFoundError(msg) from exc

    cfg = Config.fromfile(args.config)
    cfg.work_dir = osp.join("./work_dirs",
                            osp.splitext(osp.basename(args.config))[0])

    if args.colossalai:
        cfg.strategy = None
        cfg.pop("runner_type")

    # build the runner from config
    runner = (
        Runner.from_cfg(cfg)
        if "runner_type" not in cfg else RUNNERS.build(cfg))

    if args.colossalai:
        from colossalai.checkpoint_io import GeneralCheckpointIO
        GeneralCheckpointIO().load_sharded_model(runner.model, Path(args.in_file))
    else:
        state_dict = torch.load(args.in_file)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        runner.model.load_state_dict(state_dict, strict=False)

    process_checkpoint(runner, args.out_dir, args.save_keys)


if __name__ == "__main__":
    main()
