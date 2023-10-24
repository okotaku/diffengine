from torch import nn


class BaseLoss(nn.Module):
    """Base class for all losses."""

    @property
    def use_snr(self) -> bool:
        """Whether or not this loss uses SNR."""
        return False
