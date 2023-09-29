# https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/resampler.py
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin


def get_ffn(embed_dims, ffn_ratio=4):
    inner_dim = int(embed_dims * ffn_ratio)
    return nn.Sequential(
        nn.LayerNorm(embed_dims),
        nn.Linear(embed_dims, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, embed_dims, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) -->
    # (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) -->
    # (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    """PerceiverAttention of IP-Adapter Plus.

    Args:
        embed_dims (int): The feature dimension.
        head_dims (int): The number of head channels. Defaults to 64.
        num_heads (int): Parallel attention heads. Defaults to 16.
    """

    def __init__(self, *, embed_dims: int, head_dims=64, num_heads: int = 16):
        super().__init__()
        self.scale = head_dims**-0.5
        self.head_dims = head_dims
        self.num_heads = num_heads
        inner_dim = head_dims * num_heads

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

        self.to_q = nn.Linear(embed_dims, inner_dim, bias=False)
        self.to_kv = nn.Linear(embed_dims, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, embed_dims, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.num_heads)
        k = reshape_tensor(k, self.num_heads)
        v = reshape_tensor(v, self.num_heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.head_dims))
        # More stable with f16 than dividing afterwards
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(ModelMixin, ConfigMixin):
    """Resampler of IP-Adapter Plus.

    Args:
        embed_dims (int): The feature dimension. Defaults to 768.
        output_dims (int): The number of output channels, that is the same
            number of the channels in the
            `unet.config.cross_attention_dim`. Defaults to 1024.
        hidden_dims (int): The number of hidden channels. Defaults to 1280.
        depth (int): The number of blocks. Defaults to 8.
        head_dims (int): The number of head channels. Defaults to 64.
        num_heads (int): Parallel attention heads. Defaults to 16.
        num_queries (int): The number of queries. Defaults to 8.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
    """

    def __init__(
        self,
        embed_dims: int = 768,
        output_dims: int = 1024,
        hidden_dims: int = 1280,
        depth: int = 8,
        head_dims: int = 64,
        num_heads: int = 16,
        num_queries: int = 8,
        ffn_ratio: float = 4,
    ) -> None:
        super().__init__()

        self.latents = nn.Parameter(
            torch.randn(1, num_queries, hidden_dims) / hidden_dims**0.5)

        self.proj_in = nn.Linear(embed_dims, hidden_dims)

        self.proj_out = nn.Linear(hidden_dims, output_dims)
        self.norm_out = nn.LayerNorm(output_dims)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(
                        embed_dims=hidden_dims,
                        head_dims=head_dims,
                        num_heads=num_heads),
                    get_ffn(embed_dims=hidden_dims, ffn_ratio=ffn_ratio),
                ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)
