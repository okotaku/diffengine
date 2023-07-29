from collections import OrderedDict

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class SDCheckpointHook(Hook):
    """Delete 'vae' from checkpoint for efficient save."""
    priority = 'VERY_LOW'

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        """
        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            checkpoint (dict): Model's checkpoint.
        """
        new_ckpt = OrderedDict()
        sd_keys = checkpoint['state_dict'].keys()
        for k in sd_keys:
            if k.startswith(tuple(['unet', 'text_encoder'])):
                # if not finetune text_encoder, then not save
                if k.startswith('text_encoder') and not checkpoint[
                        'state_dict'][k].requires_grad:
                    continue
                new_ckpt[k] = checkpoint['state_dict'][k]
        checkpoint['state_dict'] = new_ckpt
