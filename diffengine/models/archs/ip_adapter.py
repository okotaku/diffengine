from collections import OrderedDict

import torch
import torch.nn.functional as F  # noqa
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
)
from diffusers.models.embeddings import ImageProjection, Resampler
from diffusers.utils import _get_model_file
from safetensors import safe_open
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


def load_ip_adapter(  # noqa: PLR0915, C901, PLR0912
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
        for k in attn_proc.state_dict():
            value_dict.update({f"{k}": state_dict["ip_adapter"][f"{key_id}.{k}"]})

        attn_proc.load_state_dict(value_dict)
        key_id += 2

    if "proj.weight" in state_dict["image_proj"]:
        # IP-Adapter
        image_proj_state_dict = {}
        image_proj_state_dict.update(
            {
                "image_embeds.weight": state_dict["image_proj"]["proj.weight"],
                "image_embeds.bias": state_dict["image_proj"]["proj.bias"],
                "norm.weight": state_dict["image_proj"]["norm.weight"],
                "norm.bias": state_dict["image_proj"]["norm.bias"],
            },
        )
        image_projection.load_state_dict(image_proj_state_dict)
        del image_proj_state_dict
    elif "proj.3.weight" in state_dict["image_proj"]:
        image_proj_state_dict = {}
        image_proj_state_dict.update(
            {
                "ff.net.0.proj.weight": state_dict["image_proj"]["proj.0.weight"],
                "ff.net.0.proj.bias": state_dict["image_proj"]["proj.0.bias"],
                "ff.net.2.weight": state_dict["image_proj"]["proj.2.weight"],
                "ff.net.2.bias": state_dict["image_proj"]["proj.2.bias"],
                "norm.weight": state_dict["image_proj"]["proj.3.weight"],
                "norm.bias": state_dict["image_proj"]["proj.3.bias"],
            },
        )
        image_projection.load_state_dict(image_proj_state_dict)
        del image_proj_state_dict
    else:
        # IP-Adapter Plus
        new_sd = OrderedDict()
        for k, v in state_dict["image_proj"].items():
            if "0.to" in k:
                new_k = k.replace("0.to", "2.to")
            elif "1.0.weight" in k:
                new_k = k.replace("1.0.weight", "3.0.weight")
            elif "1.0.bias" in k:
                new_k = k.replace("1.0.bias", "3.0.bias")
            elif "1.1.weight" in k:
                new_k = k.replace("1.1.weight", "3.1.net.0.proj.weight")
            elif "1.3.weight" in k:
                new_k = k.replace("1.3.weight", "3.1.net.2.weight")
            else:
                new_k = k

            if "norm1" in new_k:
                new_sd[new_k.replace("0.norm1", "0")] = v
            elif "norm2" in new_k:
                new_sd[new_k.replace("0.norm2", "1")] = v
            elif "to_kv" in new_k:
                v_chunk = v.chunk(2, dim=0)
                new_sd[new_k.replace("to_kv", "to_k")] = v_chunk[0]
                new_sd[new_k.replace("to_kv", "to_v")] = v_chunk[1]
            elif "to_out" in new_k:
                new_sd[new_k.replace("to_out", "to_out.0")] = v
            else:
                new_sd[new_k] = v
        image_projection.load_state_dict(new_sd)
    del state_dict
    torch.cuda.empty_cache()


def process_ip_adapter_state_dict(  # noqa: PLR0915, C901, PLR0912
    unet: nn.Module, image_projection: nn.Module) -> dict:
    """Process IP-Adapter state dict."""
    adapter_modules = torch.nn.ModuleList([
        v if isinstance(v, nn.Module) else nn.Identity(
            ) for v in unet.attn_processors.values()])

    # not save no grad key
    ip_image_projection_state_dict = OrderedDict()
    if isinstance(image_projection, ImageProjection):
        for k, v in image_projection.state_dict().items():
            new_k = k.replace("image_embeds.", "proj.")
            ip_image_projection_state_dict[new_k] = v
    elif isinstance(image_projection, Resampler):
        for k, v in image_projection.state_dict().items():
            if "2.to" in k:
                new_k = k.replace("2.to", "0.to")
            elif "layers.3.0.weight" in k:
                new_k = k.replace("layers.3.0.weight", "layers.3.0.norm1.weight")
            elif "layers.3.0.bias" in k:
                new_k = k.replace("layers.3.0.bias", "layers.3.0.norm1.bias")
            elif "layers.3.1.weight" in k:
                new_k = k.replace("layers.3.1.weight", "layers.3.0.norm2.weight")
            elif "layers.3.1.bias" in k:
                new_k = k.replace("layers.3.1.bias", "layers.3.0.norm2.bias")
            elif "3.0.weight" in k:
                new_k = k.replace("3.0.weight", "1.0.weight")
            elif "3.0.bias" in k:
                new_k = k.replace("3.0.bias", "1.0.bias")
            elif "3.0.weight" in k:
                new_k = k.replace("3.0.weight", "1.0.weight")
            elif "3.1.net.0.proj.weight" in k:
                new_k = k.replace("3.1.net.0.proj.weight", "1.1.weight")
            elif "3.1.net.2.weight" in k:
                new_k = k.replace("3.1.net.2.weight", "1.3.weight")
            elif "layers.0.0" in k:
                new_k = k.replace("layers.0.0", "layers.0.0.norm1")
            elif "layers.0.1" in k:
                new_k = k.replace("layers.0.1", "layers.0.0.norm2")
            elif "layers.1.0" in k:
                new_k = k.replace("layers.1.0", "layers.1.0.norm1")
            elif "layers.1.1" in k:
                new_k = k.replace("layers.1.1", "layers.1.0.norm2")
            elif "layers.2.0" in k:
                new_k = k.replace("layers.2.0", "layers.2.0.norm1")
            elif "layers.2.1" in k:
                new_k = k.replace("layers.2.1", "layers.2.0.norm2")
            else:
                new_k = k

            if "norm_cross" in new_k:
                ip_image_projection_state_dict[new_k.replace("norm_cross", "norm1")] = v
            elif "layer_norm" in new_k:
                ip_image_projection_state_dict[new_k.replace("layer_norm", "norm2")] = v
            elif "to_k" in new_k:
                ip_image_projection_state_dict[
                    new_k.replace("to_k", "to_kv")] = torch.cat([
                    v, image_projection.state_dict()[k.replace("to_k", "to_v")]], dim=0)
            elif "to_v" in new_k:
                continue
            elif "to_out.0" in new_k:
                ip_image_projection_state_dict[new_k.replace("to_out.0", "to_out")] = v
            else:
                ip_image_projection_state_dict[new_k] = v

    return {"image_proj": ip_image_projection_state_dict,
            "ip_adapter": adapter_modules.state_dict()}
