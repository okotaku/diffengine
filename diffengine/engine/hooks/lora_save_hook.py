import os.path as osp
from collections import OrderedDict

from diffusers.loaders import text_encoder_lora_state_dict  # noqa
from diffusers.loaders import LoraLoaderMixin
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS

from diffengine.models.archs import unet_attn_processors_state_dict


@HOOKS.register_module()
class LoRASaveHook(Hook):
    """Save LoRA weights with diffusers format and pick up LoRA weights from
    checkpoint."""
    priority = 'VERY_LOW'

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        """
        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            checkpoint (dict): Model's checkpoint.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        unet_lora_layers_to_save = unet_attn_processors_state_dict(model.unet)
        # text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(
        #   model.text_encoder)
        ckpt_path = osp.join(runner.work_dir, f'step{runner.iter}')
        LoraLoaderMixin.save_lora_weights(
            ckpt_path,
            unet_lora_layers=unet_lora_layers_to_save,
            # text_encoder_lora_layers=text_encoder_lora_layers_to_save,
        )

        # not save no grad key
        new_ckpt = OrderedDict()
        sd_keys = checkpoint['state_dict'].keys()
        for k in sd_keys:
            if '.processor.' in k:
                new_ckpt[k] = checkpoint['state_dict'][k]
        checkpoint['state_dict'] = new_ckpt
