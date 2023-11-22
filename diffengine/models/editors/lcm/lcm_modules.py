# copied from https://github.com/huggingface/diffusers/blob/main/examples/
#    consistency_distillation/train_lcm_distill_sdxl_wds.py
import torch
from torch import nn


def extract_into_tensor(x: torch.Tensor, timesteps: torch.Tensor,
                        ) -> torch.Tensor:
    """Extract time-dependent values from a tensor."""
    b = timesteps.shape[0]
    out = x.gather(-1, timesteps)
    return out.reshape(b, 1, 1, 1)


def scalings_for_boundary_conditions(
    timestep: torch.Tensor, sigma_data: float=0.5) -> tuple:
    """Scalings for boundary conditions.

    From LCMScheduler.get_scalings_for_boundary_condition_discrete
    """
    b = timestep.shape[0]
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip.reshape(b, 1, 1, 1), c_out.reshape(b, 1, 1, 1)


class DDIMSolver(nn.Module):
    """DDIM solver."""

    def __init__(self, alpha_cumprods: torch.Tensor,
                 timesteps: int = 1000,
                 ddim_timesteps: int = 50) -> None:
        super().__init__()
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps

        ddim_timesteps_tensor = (torch.arange(1, ddim_timesteps + 1) * step_ratio) - 1
        ddim_alpha_cumprods = alpha_cumprods[ddim_timesteps_tensor]
        ddim_alpha_cumprods_prev = torch.cat([
            alpha_cumprods[:1], alpha_cumprods[ddim_timesteps_tensor[:-1]]])

        # convert to torch tensors
        self.register_buffer("ddim_timesteps", ddim_timesteps_tensor.long())
        self.register_buffer("ddim_alpha_cumprods", ddim_alpha_cumprods)
        self.register_buffer("ddim_alpha_cumprods_prev",
                             ddim_alpha_cumprods_prev)

    def ddim_step(self, pred_x0: torch.Tensor, pred_noise: torch.Tensor,
                  timestep_index: torch.Tensor) -> torch.Tensor:
        """DDIM step."""
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev.to(pred_x0.dtype)
