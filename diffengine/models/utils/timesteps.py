import torch
from diffusers import DDPMScheduler
from torch import nn

from diffengine.registry import MODELS


@MODELS.register_module()
class TimeSteps(nn.Module):
    """Time Steps module."""

    def forward(self, scheduler: DDPMScheduler, num_batches: int, device: str,
                ) -> torch.Tensor:
        """Forward pass.

        Generates time steps for the given batches.

        Args:
        ----
            scheduler (DDPMScheduler): Scheduler for training diffusion model.
            num_batches (int): Batch size.
            device (str): Device.
        """
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps, (num_batches, ),
            device=device)
        return timesteps.long()



@MODELS.register_module()
class LaterTimeSteps(nn.Module):
    """Later biased Time Steps module.

    Args:
    ----
        bias_multiplier (float): Bias multiplier. Defaults to 10.
        bias_portion (float): Portion of later time steps to bias.
            Defaults to 0.25.
    """

    def __init__(self, bias_multiplier: float = 5., bias_portion: float = 0.25,
                 ) -> None:
        super().__init__()
        lower_limit = 0.
        upper_limit = 1.
        assert lower_limit <= bias_portion <= upper_limit, \
            "bias_portion must be in [0, 1]"

        self.bias_multiplier = bias_multiplier
        self.bias_portion = bias_portion

    def forward(self, scheduler: DDPMScheduler, num_batches: int, device: str,
                ) -> torch.Tensor:
        """Forward pass.

        Generates time steps for the given batches.

        Args:
        ----
            scheduler (DDPMScheduler): Scheduler for training diffusion model.
            num_batches (int): Batch size.
            device (str): Device.
        """
        weights = torch.ones(
            scheduler.config.num_train_timesteps, device=device)
        num_to_bias = int(
            self.bias_portion * scheduler.config.num_train_timesteps)
        bias_indices = slice(-num_to_bias, None)
        weights[bias_indices] *= self.bias_multiplier
        weights /= weights.sum()

        timesteps = torch.multinomial(weights, num_batches, replacement=True)
        return timesteps.long()


@MODELS.register_module()
class EarlierTimeSteps(nn.Module):
    """Earlier biased Time Steps module.

    Args:
    ----
        bias_multiplier (float): Bias multiplier. Defaults to 10.
        bias_portion (float): Portion of earlier time steps to bias.
            Defaults to 0.25.
    """

    def __init__(self, bias_multiplier: float = 5., bias_portion: float = 0.25,
                 ) -> None:
        super().__init__()
        lower_limit = 0.
        upper_limit = 1.
        assert lower_limit <= bias_portion <= upper_limit, \
            "bias_portion must be in [0, 1]"

        self.bias_multiplier = bias_multiplier
        self.bias_portion = bias_portion

    def forward(self, scheduler: DDPMScheduler, num_batches: int, device: str,
                ) -> torch.Tensor:
        """Forward pass.

        Generates time steps for the given batches.

        Args:
        ----
            scheduler (DDPMScheduler): Scheduler for training diffusion model.
            num_batches (int): Batch size.
            device (str): Device.
        """
        weights = torch.ones(
            scheduler.config.num_train_timesteps, device=device)
        num_to_bias = int(
            self.bias_portion * scheduler.config.num_train_timesteps)
        bias_indices = slice(0, num_to_bias)
        weights[bias_indices] *= self.bias_multiplier
        weights /= weights.sum()

        timesteps = torch.multinomial(weights, num_batches, replacement=True)
        return timesteps.long()


@MODELS.register_module()
class RangeTimeSteps(nn.Module):
    """Range biased Time Steps module.

    Args:
    ----
        bias_multiplier (float): Bias multiplier. Defaults to 10.
        bias_begin (float): Portion of begin time steps to bias.
            Defaults to 0.25.
        bias_end (float): Portion of end time steps to bias.
            Defaults to 0.75.
    """

    def __init__(self, bias_multiplier: float = 5., bias_begin: float = 0.25,
                 bias_end: float = 0.75) -> None:
        super().__init__()
        lower_limit = 0.
        upper_limit = 1.
        assert bias_begin < bias_end, "bias_begin must be less than bias_end"
        assert lower_limit <= bias_begin <= upper_limit, \
            "bias_begin must be in [0, 1]"
        assert lower_limit <= bias_end <= upper_limit, \
            "bias_end must be in [0, 1]"

        self.bias_multiplier = bias_multiplier
        self.bias_begin = bias_begin
        self.bias_end = bias_end

    def forward(self, scheduler: DDPMScheduler, num_batches: int, device: str,
                ) -> torch.Tensor:
        """Forward pass.

        Generates time steps for the given batches.

        Args:
        ----
            scheduler (DDPMScheduler): Scheduler for training diffusion model.
            num_batches (int): Batch size.
            device (str): Device.
        """
        weights = torch.ones(
            scheduler.config.num_train_timesteps, device=device)
        bias_begin = int(
            self.bias_begin * scheduler.config.num_train_timesteps)
        bias_end = int(
            self.bias_end * scheduler.config.num_train_timesteps)
        bias_indices = slice(bias_begin, bias_end)
        weights[bias_indices] *= self.bias_multiplier
        weights /= weights.sum()

        timesteps = torch.multinomial(weights, num_batches, replacement=True)
        return timesteps.long()


@MODELS.register_module()
class CubicSamplingTimeSteps(nn.Module):
    """Cubic Sampling Time Steps module.

    For more details about why cubic sampling is used, refer to section
    3.4 of https://arxiv.org/abs/2302.08453
    """

    def forward(self, scheduler: DDPMScheduler, num_batches: int, device: str,
                ) -> torch.Tensor:
        """Forward pass.

        Generates time steps for the given batches.

        Args:
        ----
            scheduler (DDPMScheduler): Scheduler for training diffusion model.
            num_batches (int): Batch size.
            device (str): Device.
        """
        timesteps = torch.rand((num_batches, ), device=device)
        timesteps = (
            1 - timesteps ** 3) * scheduler.config.num_train_timesteps
        timesteps = timesteps.long()
        return timesteps.clamp(
            0, scheduler.config.num_train_timesteps - 1)


@MODELS.register_module()
class WuerstchenRandomTimeSteps(nn.Module):
    """Wuerstchen Random Time Steps module."""

    def forward(self, num_batches: int, device: str,
                ) -> torch.Tensor:
        """Forward pass.

        Generates time steps for the given batches.

        Args:
        ----
            scheduler (DDPMScheduler): Scheduler for training diffusion model.
            num_batches (int): Batch size.
            device (str): Device.
        """
        return torch.rand((num_batches, ), device=device)
