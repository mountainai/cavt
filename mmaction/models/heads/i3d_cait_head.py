import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead

@HEADS.register_module()
class I3DCaitHead(BaseHead):
    """Classification head for Cait I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='MeanSquaredLoss')
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='MeanSquaredLoss'),
                 init_std=0.02,
                 dropout_ratio=0.5,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.init_std = init_std
        self.layernorm = nn.LayerNorm(in_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_ratio)
        self.fc_cls = nn.Linear(self.in_channels, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The regression scores for input samples.
        """
        # [N, 1, in_channels]
        x = x.view(x.shape[0], -1)
        x = self.relu(x)
        x = self.dropout(x)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        return cls_score