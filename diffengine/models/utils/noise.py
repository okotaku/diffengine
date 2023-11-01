import random

import torch
from torch import nn

from diffengine.registry import MODELS


@MODELS.register_module()
class WhiteNoise(nn.Module):
    """White noise module."""

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Generates noise for the given latents.

        Args:
        ----
            latents (torch.Tensor): Latent vectors.
        """
        return torch.randn_like(latents)


@MODELS.register_module()
class OffsetNoise(nn.Module):
    """Offset noise module.

    https://www.crosslabs.org/blog/diffusion-with-offset-noise

    Args:
    ----
        offset_weight (float): Noise offset weight. Defaults to 0.05.
    """

    def __init__(self, offset_weight: float = 0.05) -> None:
        super().__init__()
        self.offset_weight = offset_weight

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Generates noise for the given latents.

        Args:
        ----
            latents (torch.Tensor): Latent vectors.
        """
        noise = torch.randn_like(latents)
        return noise + self.offset_weight * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=noise.device)


@MODELS.register_module()
class PyramidNoise(nn.Module):
    """Pyramid noise module.

    https://wandb.ai/johnowhitaker/multires_noise/reports/
    Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2

    Args:
    ----
        discount (float): Noise offset weight. Defaults to 0.9.
        random_multiplier (bool): Whether to use random multiplier.
            Defaults to True.
    """

    def __init__(self, discount: float = 0.9,
                 *,
                 random_multiplier: bool = True) -> None:
        super().__init__()
        self.discount = discount
        self.random_multiplier = random_multiplier

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Generates noise for the given latents.

        Args:
        ----
            latents (torch.Tensor): Latent vectors.
        """
        noise = torch.randn_like(latents)

        b, c, w, h = latents.shape
        u = nn.Upsample(size=(w, h), mode="bilinear")
        for i in range(16):
            r = random.random() * 2 + 2 if self.random_multiplier else 2  # noqa: S311

            w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
            noise += u(
                torch.randn(b, c, w, h).to(latents)) * self.discount ** i
            if w==1 or h==1:
                break

        return noise / noise.std()
