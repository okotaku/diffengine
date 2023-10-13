from typing import Optional

import torch
import torch.nn.functional as F  # noqa
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
)
from torch import nn


class IPAttnProcessor(nn.Module):
    """Attention processor for IP-Adapater.

    Args:
        hidden_size (int):
            The hidden size of the attention layer.
        cross_attention_dim (int, optional):
            The number of channels in the `encoder_hidden_states`.
            Defaults to None.
        text_context_len (int):
            The context length of the text features. Defaults to 77.
    """

    def __init__(self,
                 hidden_size: int,
                 cross_attention_dim: Optional[int] = None,
                 text_context_len: int = 77) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.text_context_len = text_context_len

        self.to_k_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        image_input_ndim = 4

        if input_ndim == image_input_ndim:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel,
                                               height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None else encoder_hidden_states.shape)
        attention_mask = attn.prepare_attention_mask(attention_mask,
                                                     sequence_length,
                                                     batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(
                1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states)

        # split hidden states
        encoder_hidden_states, ip_hidden_states = (
            encoder_hidden_states[:, :self.text_context_len, :],
            encoder_hidden_states[:, self.text_context_len:, :],
        )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        hidden_states = hidden_states + scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == image_input_ndim:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


class IPAttnProcessor2_0(nn.Module):  # noqa
    """Attention processor for IP-Adapater for PyTorch 2.0.

    Args:
        hidden_size (int):
            The hidden size of the attention layer.
        cross_attention_dim (int, optional):
            The number of channels in the `encoder_hidden_states`.
            Defaults to None.
        text_context_len (int):
            The context length of the text features. Defaults to 77.
    """

    def __init__(self,
                 hidden_size: int,
                 cross_attention_dim: Optional[int] = None,
                 text_context_len: int = 77) -> None:
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            msg = ("AttnProcessor2_0 requires PyTorch 2.0, to use it,"
                   " please upgrade PyTorch to 2.0.")
            raise ImportError(msg)

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.text_context_len = text_context_len

        self.to_k_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        image_input_ndim = 4

        if input_ndim == image_input_ndim:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel,
                                               height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None else encoder_hidden_states.shape)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1,
                                                 attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(
                1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states)

        # split hidden states
        encoder_hidden_states, ip_hidden_states = (
            encoder_hidden_states[:, :self.text_context_len, :],
            encoder_hidden_states[:, self.text_context_len:, :],
        )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO(takuoko): add support for attn.scale when we move to Torch 2.1  # noqa
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = ip_key.view(batch_size, -1, attn.heads,
                             head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads,
                                 head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO(takuoko): add support for attn.scale when we move to Torch 2.1  # noqa
        ip_hidden_states = F.scaled_dot_product_attention(
            query,
            ip_key,
            ip_value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False)

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        hidden_states = hidden_states + scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == image_input_ndim:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


class CNAttnProcessor:
    """Default processor for performing attention-related computations.

    Args:
        clip_extra_context_tokens (int): The number of expansion ratio of proj
            network hidden layer channels Defaults to 4.
    """

    def __init__(self, clip_extra_context_tokens: int = 4) -> None:
        self.clip_extra_context_tokens = clip_extra_context_tokens

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            scale: float = 1.0,  # noqa
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        image_input_ndim = 4

        if input_ndim == image_input_ndim:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel,
                                               height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None else encoder_hidden_states.shape)
        attention_mask = attn.prepare_attention_mask(attention_mask,
                                                     sequence_length,
                                                     batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(
                1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            # only use text
            encoder_hidden_states = (
                encoder_hidden_states[:, :self.clip_extra_context_tokens])
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == image_input_ndim:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


class CNAttnProcessor2_0:  # noqa
    """Processor for implementing scaled dot-product attention (enabled by
    default if you're using PyTorch 2.0).

    Args:
        clip_extra_context_tokens (int): The number of expansion ratio of proj
            network hidden layer channels Defaults to 4.
    """

    def __init__(self, clip_extra_context_tokens: int = 4) -> None:
        if not hasattr(F, "scaled_dot_product_attention"):
            msg = ("AttnProcessor2_0 requires PyTorch 2.0, to use it,"
                   " please upgrade PyTorch to 2.0.")
            raise ImportError(msg)
        self.clip_extra_context_tokens = clip_extra_context_tokens

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            scale: float = 1.0,  # noqa
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        image_input_ndim = 4

        if input_ndim == image_input_ndim:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel,
                                               height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None else encoder_hidden_states.shape)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1,
                                                 attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(
                1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = (
                encoder_hidden_states[:, :self.clip_extra_context_tokens])
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO(takuoko): add support for attn.scale when we move to Torch 2.1  # noqa
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == image_input_ndim:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


def set_unet_ip_adapter(unet: nn.Module) -> None:
    """Set IP-Adapter for Unet.

    Args:
        unet (nn.Module): The unet to set IP-Adapter.
    """
    attn_procs = {}
    for name in unet.attn_processors:
        cross_attention_dim = None if name.endswith(
            "attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(
                unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if cross_attention_dim is None:
            attn_processor_class = (
                AttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention")
                else AttnProcessor)
            attn_procs[name] = attn_processor_class()
        else:
            attn_processor_class = (
                IPAttnProcessor2_0 if hasattr(
                    F, "scaled_dot_product_attention") else IPAttnProcessor)
            attn_procs[name] = attn_processor_class(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim)
    unet.set_attn_processor(attn_procs)


def set_controlnet_ip_adapter(controlnet, clip_extra_context_tokens: int = 4):
    """Set IP-Adapter for Unet.

    Args:
        controlnet (nn.Module): The ControlNet to set IP-Adapter.
        clip_extra_context_tokens (int): The number of expansion ratio of proj
            network hidden layer channels Defaults to 4.
    """
    attn_processor_class = (
        CNAttnProcessor2_0
        if hasattr(F, "scaled_dot_product_attention") else CNAttnProcessor)
    controlnet.set_attn_processor(
        attn_processor_class(
            clip_extra_context_tokens=clip_extra_context_tokens))
