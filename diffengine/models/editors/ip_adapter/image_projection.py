import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from torch import nn


class ImageProjModel(ModelMixin, ConfigMixin):
    """Projection Model.

    Args:
        cross_attention_dim (int): The number of channels in the
            `unet.config.cross_attention_dim`. Defaults to 1024.
        clip_embeddings_dim (int): The number of channels in the
            `image_encoder.config.projection_dim`. Defaults to 1024.
        clip_extra_context_tokens (int): The number of expansion ratio of proj
            network hidden layer channels Defaults to 4.
    """

    def __init__(self,
                 cross_attention_dim: int = 1024,
                 clip_embeddings_dim: int = 1024,
                 clip_extra_context_tokens: int = 4) -> None:
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(
            clip_embeddings_dim,
            self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim)
        return self.norm(clip_extra_context_tokens)
