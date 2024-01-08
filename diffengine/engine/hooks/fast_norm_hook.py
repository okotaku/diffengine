import torch
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from torch import nn
from torch.nn import functional as F  # noqa

try:
    import apex
except ImportError:
    apex = None


def _fast_gn_forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa
    """Faster group normalization forward.

    Copied from
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/
    fast_norm.py
    """
    if torch.is_autocast_enabled():
        dt = torch.get_autocast_gpu_dtype()
        x = x.to(dt)
        weight = self.weight.to(dt)
        bias = self.bias.to(dt) if self.bias is not None else None
    else:
        weight = self.weight
        bias = self.bias

    with torch.cuda.amp.autocast(enabled=False):
        return F.group_norm(x, self.num_groups, weight, bias, self.eps)


@HOOKS.register_module()
class FastNormHook(Hook):
    """Fast Normalization Hook.

    Replace the normalization layer with a faster one.

    Args:
    ----
        fuse_text_encoder_ln (bool): Whether to fuse the text encoder layer
            normalization. Defaults to False.
        fuse_main_ln (bool): Whether to replace the layer normalization
            in main module like unet or transformer. Defaults to True.
        fuse_gn (bool) : Whether to replace the group normalization.
            Defaults to False.
    """

    priority = "VERY_LOW"

    def __init__(self, *, fuse_text_encoder_ln: bool = False,
                 fuse_main_ln: bool = True,
                 fuse_gn: bool = False) -> None:
        super().__init__()
        if apex is None:
            msg = "Please install apex to use FastNormHook."
            raise ImportError(
                msg)
        self.fuse_text_encoder_ln = fuse_text_encoder_ln
        self.fuse_main_ln = fuse_main_ln
        self.fuse_gn = fuse_gn

    def _replace_ln(self, module: nn.Module, name: str, device: str) -> None:
        """Replace the layer normalization with a fused one."""
        from apex.normalization import FusedLayerNorm
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if isinstance(target_attr, torch.nn.LayerNorm):
                print_log(f"replaced LN: {name}")
                normalized_shape = target_attr.normalized_shape
                eps = target_attr.eps
                elementwise_affine = target_attr.elementwise_affine
                # Create a new fused layer normalization with the same arguments
                fused_ln = FusedLayerNorm(normalized_shape, eps, elementwise_affine)
                fused_ln.load_state_dict(target_attr.state_dict())
                fused_ln.to(device)
                setattr(module, attr_str, fused_ln)

        for name, immediate_child_module in module.named_children():
            self._replace_ln(immediate_child_module, name, device)

    def _replace_gn(self, module: nn.Module, name: str, device: str) -> None:
        """Replace the layer normalization with a fused one."""
        from apex.contrib.group_norm import GroupNorm
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if isinstance(target_attr, torch.nn.GroupNorm):
                print_log(f"replaced GN: {name}")
                num_groups = target_attr.num_groups
                num_channels = target_attr.num_channels
                eps = target_attr.eps
                affine = target_attr.affine
                # Create a new fused layer normalization with the same arguments
                fused_gn = GroupNorm(num_groups, num_channels, eps, affine)
                fused_gn.load_state_dict(target_attr.state_dict())
                fused_gn.to(device)
                setattr(module, attr_str, fused_gn)

        for name, immediate_child_module in module.named_children():
            self._replace_gn(immediate_child_module, name, device)

    def _replace_gn_forward(self, module: nn.Module, name: str) -> None:
        """Replace the group normalization forward with a faster one."""
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if isinstance(target_attr, torch.nn.GroupNorm):
                print_log(f"replaced GN Forward: {name}")
                target_attr.forward = _fast_gn_forward.__get__(
                    target_attr, torch.nn.GroupNorm)

        for name, immediate_child_module in module.named_children():
            self._replace_gn_forward(immediate_child_module, name)

    def before_train(self, runner: Runner) -> None:  # noqa: C901 PLR0912
        """Replace the normalization layer with a faster one.

        Args:
        ----
            runner (Runner): The runner of the training process.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if self.fuse_main_ln:
            if hasattr(model, "unet"):
                self._replace_ln(model.unet, "model", model.device)
            elif hasattr(model, "transformer"):
                self._replace_ln(model.transformer, "model", model.device)
            if hasattr(model, "controlnet"):
                self._replace_ln(model.controlnet, "model", model.device)

        if self.fuse_gn:
            if hasattr(model, "unet"):
                self._replace_gn(model.unet, "model", model.device)
            if hasattr(model, "controlnet"):
                self._replace_gn(model.controlnet, "model", model.device)
        else:
            if hasattr(model, "unet"):
                self._replace_gn_forward(model.unet, "unet")
            if hasattr(model, "controlnet"):
                self._replace_gn_forward(model.controlnet, "unet")

        if self.fuse_text_encoder_ln:
            if hasattr(model, "text_encoder"):
                self._replace_ln(model.text_encoder, "model", model.device)
            if hasattr(model, "text_encoder_one"):
                self._replace_ln(model.text_encoder_one, "model", model.device)
            if hasattr(model, "text_encoder_two"):
                self._replace_ln(model.text_encoder_two, "model", model.device)
