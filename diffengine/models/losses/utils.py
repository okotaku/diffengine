import torch


def compute_snr(timesteps, alphas_cumprod) -> torch.Tensor:  # noqa
    """Compute SNR.

    Refer to https://github.com/TiankaiHang/Min-SNR-Diffusion-Tra
    ining/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussi
    an_diffusion.py#L847-L849.
    """
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
    return (alpha / sigma)**2
