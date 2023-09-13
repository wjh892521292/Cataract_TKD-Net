# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmcls.models.builder import LOSSES
from mmcls.models.losses.utils import weight_reduce_loss


def nll_loss(pred,
             target,
             weight = None,
             reduction = 'mean',
             avg_factor = None):
    r"""NLL loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The ground truth label of the prediction with
            shape (N, \*).
        weight (torch.Tensor, optional): Sample-wise loss weight with shape
            (N, ). Defaults to None.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' ,
            loss is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: Loss.
    """
    loss = F.nll_loss(pred, target, reduction = 'none')
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class NLLLoss(nn.Module):
    """NLL loss.

    Args:
        reduction (str): The method used to reduce the loss into
            a scalar. Options are "none" and "mean". Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self,
                 reduction = 'mean',
                 loss_weight = 1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight = None,
                avg_factor = None,
                reduction_override = None):
        r"""Sigmoid focal loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, \*).
            target (torch.Tensor): The ground truth label of the prediction
                with shape (N, \*), N or (N,1).
            weight (torch.Tensor, optional): Sample-wise loss weight with shape
                (N, \*). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss into a scalar. Options are "none", "mean" and "sum".
                Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * nll_loss(
            pred,
            target,
            weight,
            reduction = reduction,
            avg_factor = avg_factor)
        return loss_cls
