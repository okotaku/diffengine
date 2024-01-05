# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: S603
import logging
import os
import random
import subprocess
import sys

from mmengine.logging import print_log

import diffengine
from diffengine.tools import copy_cfg, list_cfg, train
from diffengine.tools.model_converters import publish_model2diffusers
from diffengine.tools.preprocess import bucket_ids

# Define valid modes
MODES = ("list-cfg", "copy-cfg",
         "train", "convert", "preprocess")

CLI_HELP_MSG = \
    f"""
    Arguments received: {['diffengine'] + sys.argv[1:]!s}. diffengine commands use the following syntax:

        diffengine MODE MODE_ARGS ARGS

        Where   MODE (required) is one of {MODES}
                MODE_ARG (optional) is the argument for specific mode
                ARGS (optional) are the arguments for specific command

    Some usages for diffengine commands: (See more by using -h for specific command!)

        1. List all predefined configs:
            diffengine list-cfg
        2. Copy a predefined config to a given path:
            diffengine copy-cfg $CONFIG $SAVE_FILE
        3-1. Fine-tune by a single GPU:
            diffengine train $CONFIG
        3-2. Fine-tune by multiple GPUs:
            NPROC_PER_NODE=$GPU_NUM diffengine train $CONFIG
        4-1. Convert the pth model to HuggingFace's model:
            diffengine convert pth_to_hf $CONFIG $PATH_TO_PTH_MODEL $SAVE_PATH_TO_HF_MODEL
        5-1. Preprocess bucket ids:
            diffengine preprocess bucket_ids

    Run special commands:

        diffengine help
        diffengine version

    GitHub: https://github.com/okotaku/diffengine
    """  # noqa: E501


PREPROCESS_HELP_MSG = \
    f"""
    Arguments received: {['diffengine'] + sys.argv[1:]!s}. diffengine commands use the following syntax:

        diffengine MODE MODE_ARGS ARGS

        Where   MODE (required) is one of {MODES}
                MODE_ARG (optional) is the argument for specific mode
                ARGS (optional) are the arguments for specific command

    Some usages for preprocess: (See more by using -h for specific command!)

        1. Preprocess arxiv dataset:
            diffengine preprocess bucket_ids

    GitHub: https://github.com/InternLM/diffengine
    """  # noqa: E501


special = {
    "help": lambda: print_log(CLI_HELP_MSG, "current"),
    "version": lambda: print_log(diffengine.__version__, "current"),
}
special = {
    **special,
    **{f"-{k[0]}": v
       for k, v in special.items()},
    **{f"--{k}": v
       for k, v in special.items()},
}

modes: dict = {
    "list-cfg": list_cfg.__file__,
    "copy-cfg": copy_cfg.__file__,
    "train": train.__file__,
    "convert": publish_model2diffusers.__file__,
    "preprocess": {
        "bucket_ids": bucket_ids.__file__,
        "--help": lambda: print_log(PREPROCESS_HELP_MSG, "current"),
        "-h": lambda: print_log(PREPROCESS_HELP_MSG, "current"),
    },
}


def cli() -> None:
    """CLI entry point."""
    args = sys.argv[1:]
    if not args:  # no arguments passed
        print_log(CLI_HELP_MSG, "current")
        return
    if args[0].lower() in special:
        special[args[0].lower()]()
        return
    if args[0].lower() in modes:
        try:
            module = modes[args[0].lower()]
            n_arg = 0
            while not isinstance(module, str) and not callable(module):
                n_arg += 1
                module = module[args[n_arg].lower()]
            if callable(module):
                module()
            else:
                nnodes = os.environ.get("NNODES", 1)
                nproc_per_node = os.environ.get("NPROC_PER_NODE", 1)
                if nnodes == 1 and nproc_per_node == 1:
                    subprocess.run(["python", module] + args[n_arg + 1:], check=True)
                else:
                    port = os.environ.get("PORT", None)
                    if port is None:
                        port: int = random.randint(20000, 29999)  # type: ignore[no-redef] # noqa
                        print_log(f"Use random port: {port}", "current",
                                  logging.WARNING)
                    torchrun_args = [
                        f"--nnodes={nnodes}",
                        f"--node_rank={os.environ.get('NODE_RANK', 0)}",
                        f"--nproc_per_node={nproc_per_node}",
                        f"--master_addr={os.environ.get('ADDR', '127.0.0.1')}",
                        f"--master_port={port}",
                    ]
                    subprocess.run(["torchrun"] + torchrun_args + [module] +
                                   args[n_arg + 1:] +
                                   ["--launcher", "pytorch"], check=True)
        except Exception as e:
            print_log(f"WARNING: command error: '{e}'!", "current",
                      logging.WARNING)
            print_log(CLI_HELP_MSG, "current", logging.WARNING)
            return
    else:
        print_log("WARNING: command error!", "current", logging.WARNING)
        print_log(CLI_HELP_MSG, "current", logging.WARNING)
        return
