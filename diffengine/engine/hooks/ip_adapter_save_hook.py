from collections import OrderedDict
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS
from mmengine.runner import Runner

from diffengine.models.archs import process_ip_adapter_state_dict


@HOOKS.register_module()
class IPAdapterSaveHook(Hook):
    """IP Adapter Save Hook.

    Save IP-Adapter weights with diffusers format and pick up weights from
    checkpoint.
    """

    priority = "VERY_LOW"

    def before_save_checkpoint(self, runner: Runner, checkpoint: dict) -> None:
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

        adapter_state_dict = process_ip_adapter_state_dict(
            model.unet, model.image_projection)

        # not save no grad key
        new_ckpt = OrderedDict()
        sd_keys = checkpoint["state_dict"].keys()
        for k in sd_keys:
            if ".processor." in k or k.startswith("image_projection"):
                new_ckpt[k] = checkpoint["state_dict"][k]
        torch.save(adapter_state_dict, ckpt_path / "ip_adapter.bin")

        checkpoint["state_dict"] = new_ckpt
