import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss

@LOSSES.register_module()
class MeanSquaredLoss(BaseWeightedLoss):
    """Mean squared error Loss.

    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probablity distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to MeanSquaredLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_weight=1.0, class_weight=None):
        super().__init__(loss_weight=loss_weight)
        self.class_weight = class_weight

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        loss_cls = 0
        if self.class_weight:
            for k, v in self.class_weight.items():
                if torch.sum(label == k):
                    loss_cls += v * F.mse_loss(cls_score[label==k], label[label==k], **kwargs)
        else:
            loss_cls = F.mse_loss(cls_score, label, **kwargs)
        return loss_cls
