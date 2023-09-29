from typing import Dict

import torch
import torch.nn.functional as F
# yapf: enable
from diffusers.loaders import LoraLoaderMixin
# yapf: disable
from diffusers.models.attention_processor import (AttnAddedKVProcessor,
                                                  AttnAddedKVProcessor2_0,
                                                  LoRAAttnAddedKVProcessor,
                                                  LoRAAttnProcessor,
                                                  LoRAAttnProcessor2_0,
                                                  SlicedAttnAddedKVProcessor)
from mmengine import print_log
from torch import nn


def set_unet_lora(unet: nn.Module,
                  config: dict,
                  verbose: bool = True) -> nn.Module:
    """Set LoRA for Unet.

    Args:
        unet (nn.Module): The unet to set LoRA.
        config (dict): The config dict. example. dict(rank=4)
        verbose (bool): Whether to print log. Defaults to True.
    """
    rank = config.get('rank', 4)

    unet_lora_attn_procs = {}
    unet_lora_parameters = []
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith(
            'attn1.processor') else unet.config.cross_attention_dim
        if name.startswith('mid_block'):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith('up_blocks'):
            block_id = int(name[len('up_blocks.')])
            hidden_size = list(reversed(
                unet.config.block_out_channels))[block_id]
        elif name.startswith('down_blocks'):
            block_id = int(name[len('down_blocks.')])
            hidden_size = unet.config.block_out_channels[block_id]

        if isinstance(attn_processor,
                      (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor,
                       AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = (
                LoRAAttnProcessor2_0 if hasattr(
                    F, 'scaled_dot_product_attention') else LoRAAttnProcessor)

        module = lora_attn_processor_class(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank)
        unet_lora_attn_procs[name] = module
        unet_lora_parameters.extend(module.parameters())
        if verbose:
            print_log(f'Set LoRA for \'{name}\' ', 'current')
    unet.set_attn_processor(unet_lora_attn_procs)


def set_text_encoder_lora(text_encoder: nn.Module, config: dict) -> nn.Module:
    """Set LoRA for module.

    Args:
        text_encoder (nn.Module): The text_encoder to set LoRA.
        config (dict): The config dict. example. dict(rank=4)
        verbose (bool): Whether to print log. Defaults to True.
    """
    rank = config.get('rank', 4)
    _ = LoraLoaderMixin._modify_text_encoder(
        text_encoder, dtype=torch.float32, rank=rank)


def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
    """
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        # skip 'AttnProcessor2_0'
        if hasattr(attn_processor, 'state_dict'):
            for parameter_key, parameter in attn_processor.state_dict().items():  # noqa
                attn_processors_state_dict[
                    f'{attn_processor_key}.{parameter_key}'] = parameter

    return attn_processors_state_dict
