from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffengine.registry import MODELS


def compute_snr(timesteps, alphas_cumprod):
    """Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Tra
    ining/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussi
    an_diffusion.py#L847-L849  # noqa."""
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod)**0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026  # noqa
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(
        device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[...,
                                                                      None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma)**2
    return snr


@MODELS.register_module()
class SNRL2Loss(nn.Module):
    """SNR weighting gamma L2 loss.

    Args:
        loss_weight (float): Weight of this loss item.
            Defaults to ``1.``.
        snr_gamma (float): SNR weighting gamma to be used if rebalancing the
            loss.  "More details here: https://arxiv.org/abs/2303.09556."
            Defaults to ``5.``.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'l2'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 snr_gamma: float = 5.0,
                 loss_name: str = 'snrl2') -> None:

        super(SNRL2Loss, self).__init__()
        self.loss_weight = loss_weight
        self.snr_gamma = snr_gamma
        self._loss_name = loss_name

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                timesteps: torch.Tensor,
                alphas_cumprod: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        snr = compute_snr(timesteps, alphas_cumprod)
        mse_loss_weights = (
            torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)],
                        dim=1).min(dim=1)[0] / snr)
        loss = F.mse_loss(pred, gt, reduction='none')
        loss = loss.mean(
            dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        if weight is not None:
            loss = loss * weight
        loss = loss.mean()
        return loss * self.loss_weight
