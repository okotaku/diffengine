from collections import OrderedDict
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS
from torch import nn


@HOOKS.register_module()
class IPAdapterSaveHook(Hook):
    """IP Adapter Save Hook.

    Save IP-Adapter weights with diffusers format and pick up weights from
    checkpoint.
    """

    priority = "VERY_LOW"

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        """Before save checkpoint hook.

        Args:
        ----
            runner (Runner): The runner of the training, validation or testing
                process.
            checkpoint (dict): Model's checkpoint.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        ckpt_path = Path(runner.work_dir) / f"step{runner.iter}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        adapter_modules = torch.nn.ModuleList([
            v if isinstance(v, nn.Module) else nn.Identity(
                ) for v in model.unet.attn_processors.values()])

        # not save no grad key
        new_ckpt = OrderedDict()
        proj_ckpt = OrderedDict()
        sd_keys = checkpoint["state_dict"].keys()
        for k in sd_keys:
            if k.startswith("image_projection"):
                new_k = k.replace(
                    "image_projection.", "").replace("image_embeds.", "proj.")
                proj_ckpt[new_k] = checkpoint["state_dict"][k]
            if ".processor." in k or k.startswith("image_projection"):
                new_ckpt[k] = checkpoint["state_dict"][k]
        torch.save({"image_proj": proj_ckpt,
                    "ip_adapter": adapter_modules.state_dict()},
                    ckpt_path / "ip_adapter.bin")

        checkpoint["state_dict"] = new_ckpt
