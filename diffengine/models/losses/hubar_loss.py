import torch
import torch.nn.functional as F  # noqa

from diffengine.models.losses.base import BaseLoss
from diffengine.registry import MODELS


@MODELS.register_module()
class HuberLoss(BaseLoss):
    """Huber loss.

    Args:
    ----
        delta (float, optional): Specifies the threshold at which to change
            between delta-scaled L1 and L2 loss. The value must be positive.
            Default: 1.0
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'l2'.
    """

    def __init__(self,
                 delta: float = 1.0,
                 loss_weight: float = 1.0,
                 loss_name: str = "l2") -> None:

        super().__init__()
        self.delta = delta
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                weight: torch.Tensor | None = None) -> torch.Tensor:
        """Forward function.

        Args:
        ----
            pred (torch.Tensor): The predicted tensor.
            gt (torch.Tensor): The ground truth tensor.
            weight (torch.Tensor | None, optional): The loss weight.
                Defaults to None.

        Returns:
        -------
            torch.Tensor: loss
        """
        if weight is not None:
            loss = F.huber_loss(
                pred, gt, reduction="none", delta=self.delta) * weight
            return loss.mean() * self.loss_weight

        return F.huber_loss(pred, gt, delta=self.delta) * self.loss_weight
