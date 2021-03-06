# PyTorch
import torch.nn.functional as F
import torch.nn as nn


# class myDiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(myDiceLoss, self).__init__()
#
#     def forward(self, inputs, targets, smooth=1):
#         # comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)
#
#         # flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#
#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#
#         return 1 - dice

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc