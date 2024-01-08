import copy
import logging

from mmengine.hooks.ema_hook import EMAHook
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS, MODELS
from mmengine.runner import Runner


@HOOKS.register_module()
class UnetEMAHook(EMAHook):
    """Unet EMA Hook."""

    def before_run(self, runner: Runner) -> None:
        """Create an ema copy of the model.

        Args:
        ----
            runner (Runner): The runner of the training process.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        self.src_model = model.unet
        self.ema_model = MODELS.build(
            self.ema_cfg, default_args={"model": self.src_model})

    def _swap_ema_state_dict(self, checkpoint: dict) -> None:
        """Swap the state dict values of model with ema_model."""
        model_state = checkpoint["state_dict"]
        ema_state = checkpoint["ema_state_dict"]
        for k in ema_state:
            if k[:7] == "module.":
                tmp = ema_state[k]
                # 'module.' -> 'unet.'
                ema_state[k] = model_state["unet." + k[7:]]
                model_state["unet." + k[7:]] = tmp

    def after_load_checkpoint(self, runner: Runner, checkpoint: dict) -> None:
        """Resume ema parameters from checkpoint.

        Args:
        ----
            runner (Runner): The runner of the testing process.
            checkpoint (dict): Model's checkpoint.
        """
        from mmengine.runner.checkpoint import load_state_dict
        if "ema_state_dict" in checkpoint and runner._resume:  # noqa
            # The original model parameters are actually saved in ema
            # field swap the weights back to resume ema state.
            self._swap_ema_state_dict(checkpoint)
            self.ema_model.load_state_dict(
                checkpoint["ema_state_dict"], strict=self.strict_load)

        # Support load checkpoint without ema state dict.
        else:
            if runner._resume:  # noqa
                print_log(
                    "There is no `ema_state_dict` in checkpoint. "
                    "`EMAHook` will make a copy of `state_dict` as the "
                    "initial `ema_state_dict`", "current", logging.WARNING)
            sd = copy.deepcopy(checkpoint["state_dict"])
            new_sd = {}
            for k, v in sd.items():
                if k.startswith("unet."):
                    new_sd[k[5:]] = v
            load_state_dict(
                self.ema_model.module, new_sd, strict=self.strict_load)
