import torch
import torch.nn.functional as F  # noqa

from diffengine.models.losses.base import BaseLoss
from diffengine.registry import MODELS


@MODELS.register_module()
class CrossEntropyLoss(BaseLoss):
    """CrossEntropy loss.

    Args:
    ----
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        reduction: (str): The reduction method for the loss.
            Defaults to 'mean'.
        ignore_index (int): Specifies a target value that is ignored.
            Defaults to -100.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'l2'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 reduction: str = "mean",
                 ignore_index: int = -100,
                 loss_name: str = "cross_entropy") -> None:

        super().__init__()
        assert reduction in ["mean", "none"], (
            f"reduction should be 'mean' or 'none', got {reduction}"
        )
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.ignore_index = ignore_index
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
            loss = F.cross_entropy(pred, gt,
                                   ignore_index=self.ignore_index,
                                   reduction="none") * weight
            if self.reduction == "mean":
                loss = loss.mean()
            return loss * self.loss_weight

        return F.cross_entropy(pred, gt,
                               ignore_index=self.ignore_index,
                               reduction=self.reduction) * self.loss_weight
