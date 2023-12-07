import torch.nn.functional as F  # noqa
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
)
from torch import nn


def set_unet_ip_adapter(unet: nn.Module) -> None:
    """Set IP-Adapter for Unet.

    Args:
    ----
        unet (nn.Module): The unet to set IP-Adapter.
    """
    attn_procs = {}
    key_id = 1
    for name in unet.attn_processors:
        cross_attention_dim = None if name.endswith(
            "attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None or "motion_modules" in name:
            attn_processor_class = (
                AttnProcessor2_0 if hasattr(
                    F, "scaled_dot_product_attention") else AttnProcessor
            )
            attn_procs[name] = attn_processor_class()
        else:
            attn_processor_class = (
                IPAdapterAttnProcessor2_0 if hasattr(
                    F, "scaled_dot_product_attention",
                    ) else IPAdapterAttnProcessor
            )
            attn_procs[name] = attn_processor_class(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim, scale=1.0,
            ).to(dtype=unet.dtype, device=unet.device)

            key_id += 2
    unet.set_attn_processor(attn_procs)
