import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F  # noqa
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.utils import _get_model_file
from safetensors import safe_open
from torch import nn


class IPAttnProcessor2_0(nn.Module):  # noqa
    """Attention processor for IP-Adapater for PyTorch 2.0.

    Args:
    ----
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size: int,
                 cross_attention_dim: int | None = None,
                 scale: float = 1.0,
                 num_tokens: int = 4) -> None:
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            msg = ("AttnProcessor2_0 requires PyTorch 2.0, to use it, please "
                   "upgrade PyTorch to 2.0.")
            raise ImportError(
                msg)

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size,
                                 hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size,
                                 hidden_size, bias=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        **kwargs,  # noqa
    ) -> torch.Tensor:
        """Forward pass."""
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:  # noqa
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if (
                encoder_hidden_states is None
            )else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = ip_key.view(
            batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(
            batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        ip_hidden_states = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False,
        )
        with torch.no_grad():
            self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:  # noqa
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor



def set_unet_ip_adapter(unet: nn.Module, num_tokens: int) -> None:
    """Set IP-Adapter for Unet.

    Args:
    ----
        unet (nn.Module): The unet to set IP-Adapter.
        num_tokens (int): The number of tokens for IP-Adapter.
    """
    attn_procs = {}
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
            attn_procs[name] = AttnProcessor2_0()
        else:
            attn_procs[name] = IPAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim, num_tokens=num_tokens,
            ).to(dtype=unet.dtype, device=unet.device)

    unet.set_attn_processor(attn_procs)


def load_ip_adapter(
    unet: nn.Module,
                    image_projection: nn.Module,
                    pretrained_adapter: str,
                    subfolder: str,
                    weights_name: str) -> None:
    """Load IP-Adapter pretrained weights.

    Reference to diffusers/loaders/ip_adapter.py. and
    diffusers/loaders/unet.py.
    """
    model_file = _get_model_file(
        pretrained_adapter,
        subfolder=subfolder,
        weights_name=weights_name,
        cache_dir=None,
        force_download=False,
        resume_download=False,
        proxies=None,
        local_files_only=None,
        token=None,
        revision=None,
        user_agent={
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        })
    if weights_name.endswith(".safetensors"):
        state_dict: dict = {"image_proj": {}, "ip_adapter": {}}
        with safe_open(model_file, framework="pt", device="cpu") as f:
            for key in f:
                if key.startswith("image_proj."):
                    state_dict["image_proj"][
                        key.replace("image_proj.", "")] = f.get_tensor(key)
                elif key.startswith("ip_adapter."):
                    state_dict["ip_adapter"][
                        key.replace("ip_adapter.", "")] = f.get_tensor(key)
    else:
        state_dict = torch.load(model_file, map_location="cpu")

    key_id = 1
    for name, attn_proc in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith(
            "attn1.processor") else unet.config.cross_attention_dim
        if cross_attention_dim is None or "motion_modules" in name:
            continue
        value_dict = {}
        value_dict.update(
            {"to_k_ip.weight":
                state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"]})
        value_dict.update(
            {"to_v_ip.weight":
                state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"]})

        attn_proc.load_state_dict(value_dict)
        key_id += 2

    image_projection.load_state_dict(state_dict["image_proj"])
    del state_dict
    torch.cuda.empty_cache()


def process_ip_adapter_state_dict(
    unet: nn.Module, image_projection: nn.Module) -> dict:
    """Process IP-Adapter state dict."""
    adapter_modules = torch.nn.ModuleList([
        v if isinstance(v, nn.Module) else nn.Identity(
            ) for v in copy.deepcopy(unet.attn_processors).values()])
    adapter_state_dict = OrderedDict()
    for k, v in adapter_modules.state_dict().items():
        new_k = k.replace(".0.weight", ".weight")
        adapter_state_dict[new_k] = v

    return {"image_proj": image_projection.state_dict(),
            "ip_adapter": adapter_state_dict}
