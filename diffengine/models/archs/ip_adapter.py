import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F  # noqa
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
)
from diffusers.models.embeddings import ImageProjection, IPAdapterPlusImageProjection
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


def load_ip_adapter(  # noqa: C901, PLR0912
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
            {"to_k_ip.0.weight":
                state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"]})
        value_dict.update(
            {"to_v_ip.0.weight":
                state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"]})

        attn_proc.load_state_dict(value_dict)
        key_id += 2

    image_proj_state_dict = {}
    if "proj.weight" in state_dict["image_proj"]:
        # IP-Adapter
        for key, value in state_dict["image_proj"].items():
            diffusers_name = key.replace("proj", "image_embeds")
            image_proj_state_dict[diffusers_name] = value
    elif "proj.3.weight" in state_dict["image_proj"]:
        # IP-Adapter Full
        for key, value in state_dict["image_proj"].items():
            diffusers_name = key.replace("proj.0", "ff.net.0.proj")
            diffusers_name = diffusers_name.replace("proj.2", "ff.net.2")
            diffusers_name = diffusers_name.replace("proj.3", "norm")
            image_proj_state_dict[diffusers_name] = value
    else:
        # IP-Adapter Plus
        for key, value in state_dict["image_proj"].items():
            diffusers_name = key.replace("0.to", "2.to")
            diffusers_name = diffusers_name.replace("1.0.weight", "3.0.weight")
            diffusers_name = diffusers_name.replace("1.0.bias", "3.0.bias")
            diffusers_name = diffusers_name.replace(
                "1.1.weight", "3.1.net.0.proj.weight")
            diffusers_name = diffusers_name.replace("1.3.weight", "3.1.net.2.weight")

            if "norm1" in diffusers_name:
                image_proj_state_dict[
                    diffusers_name.replace("0.norm1", "0")] = value
            elif "norm2" in diffusers_name:
                image_proj_state_dict[
                    diffusers_name.replace("0.norm2", "1")] = value
            elif "to_kv" in diffusers_name:
                v_chunk = value.chunk(2, dim=0)
                image_proj_state_dict[
                    diffusers_name.replace("to_kv", "to_k")] = v_chunk[0]
                image_proj_state_dict[
                    diffusers_name.replace("to_kv", "to_v")] = v_chunk[1]
            elif "to_out" in diffusers_name:
                image_proj_state_dict[
                    diffusers_name.replace("to_out", "to_out.0")] = value
            else:
                image_proj_state_dict[diffusers_name] = value

    image_projection.load_state_dict(image_proj_state_dict)
    del image_proj_state_dict, state_dict
    torch.cuda.empty_cache()


def process_ip_adapter_state_dict(  # noqa: PLR0915, C901, PLR0912
    unet: nn.Module, image_projection: nn.Module) -> dict:
    """Process IP-Adapter state dict."""
    adapter_modules = torch.nn.ModuleList([
        v if isinstance(v, nn.Module) else nn.Identity(
            ) for v in copy.deepcopy(unet.attn_processors).values()])
    adapter_state_dict = OrderedDict()
    for k, v in adapter_modules.state_dict().items():
        new_k = k.replace(".0.weight", ".weight")
        adapter_state_dict[new_k] = v

    # not save no grad key
    ip_image_projection_state_dict = OrderedDict()
    if isinstance(image_projection, ImageProjection):
        for k, v in image_projection.state_dict().items():
            new_k = k.replace("image_embeds.", "proj.")
            ip_image_projection_state_dict[new_k] = v
    elif isinstance(image_projection, IPAdapterPlusImageProjection):
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
            "ip_adapter": adapter_state_dict}
