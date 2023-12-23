import os.path as osp
from collections import OrderedDict

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS


@HOOKS.register_module()
class PriorSaveHook(Hook):
    """Prior Save Hook.

    Save Prior weights with diffusers format and pick up Prior weights from
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
        ckpt_path = osp.join(runner.work_dir, f"step{runner.iter}")
        model.prior.save_pretrained(osp.join(ckpt_path, "prior"))
        if hasattr(
            model, "finetune_text_encoder") and model.finetune_text_encoder:
            model.text_encoder.save_pretrained(
                osp.join(ckpt_path, "text_encoder"))

        # not save no grad key
        new_ckpt = OrderedDict()
        sd_keys = checkpoint["state_dict"].keys()
        for k in sd_keys:
            if k.startswith("prior"):
                new_ckpt[k] = checkpoint["state_dict"][k]
            elif k.startswith("text_encoder") and hasattr(
                    model,
                    "finetune_text_encoder",
            ) and model.finetune_text_encoder:
                # if not finetune text_encoder, then not save
                new_ckpt[k] = checkpoint["state_dict"][k]
        checkpoint["state_dict"] = new_ckpt
