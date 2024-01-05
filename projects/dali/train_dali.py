# flake8: noqa: PTH122,PTH119,ISC002,E402,ANN201,D103,D101,PD901,PD011,ANN204,D105,D102,A003
import argparse
import os
import os.path as osp

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--resume", action="store_true", help="Whether to resume checkpoint.")
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="enable automatic-mixed-precision training")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher")
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join("./work_dirs",
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get("type", "OptimWrapper")
        assert optim_wrapper in ["OptimWrapper", "AmpOptimWrapper"], \
            "`--amp` is not supported custom optimizer wrapper type " \
            f"`{optim_wrapper}."
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.setdefault("loss_scale", "dynamic")

    # resume training
    if args.resume:
        cfg.resume = True
        cfg.load_from = None

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


import pandas as pd
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


@pipeline_def(enable_conditionals=True)
def sd_pipeline(rank, world_size, files):
    rng = fn.random.coin_flip(probability=0.5)

    img_raw, label = fn.readers.file(
        files=files,
        labels=list(range(len(files))),
        name="Reader", shard_id=rank,
        num_shards=world_size, random_shuffle=True)
    img = fn.decoders.image(
        img_raw, device="mixed", output_type=types.RGB)
    img = img.gpu()

    sizes = fn.shapes(img)

    resized = fn.resize(img, device="gpu", resize_shorter=1024,
                        interp_type=types.INTERP_LINEAR)
    resized = fn.flip(resized, horizontal=rng)
    sizes2 = fn.shapes(resized)
    output = fn.crop_mirror_normalize(
        resized,
        dtype=types.FLOAT,
        crop=(1024, 1024),
        device="gpu",
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5])
    return output, label, sizes, sizes2, rng


class Dummy:
    def __init__(self) -> None:
        pass

class DaliSDIterator:

    def __init__(self) -> None:

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        df = pd.read_csv("data/pokemon/file.csv")
        files = df.file_name.tolist()
        self.caption = df.text.values

        pipeline = sd_pipeline(
                                batch_size=1, num_threads=4, device_id=0,
                                rank=rank, world_size=world_size, files=files)

        self.dali_it = DALIGenericIterator(
            pipeline,
            ["jpg", "label", "sizes", "sizes2", "rng"],
            dynamic_shape=False,
            reader_name="Reader",
            auto_reset=True,
            prepare_first_batch=False,
            last_batch_policy=LastBatchPolicy.DROP)
        self.dataset = Dummy()

    def __next__(self):
        data = self.dali_it.__next__()
        crop_top_left = (data[0]["sizes2"][:, :2] - 1024) / 2
        time_ids = torch.cat([
            data[0]["sizes"][:, :2],
            crop_top_left,
            data[0]["sizes2"][:, :2],
        ], dim=1)
        return dict(inputs=dict(img=data[0]["jpg"],
                                text=self.caption[data[0]["label"].reshape(-1)],
                                time_ids=time_ids))

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self.caption)


def main() -> None:
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    cfg.model.data_preprocessor = dict(type="SDXLDALIDataPreprocessor")

    # build the runner from config
    train_loader = DaliSDIterator()
    runner = Runner(
        model=cfg.model,
        train_dataloader=train_loader,
        optim_wrapper=cfg.optim_wrapper,
        train_cfg=cfg.train_cfg,
        launcher=args.launcher,
        work_dir=cfg.work_dir,
        default_hooks=cfg.default_hooks,
        custom_hooks=cfg.custom_hooks,
        default_scope=cfg.default_scope,
        env_cfg=cfg.env_cfg,
    )

    # start training
    runner.train()


if __name__ == "__main__":
    main()
