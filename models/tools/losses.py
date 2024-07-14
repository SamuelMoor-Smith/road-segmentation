import torch
from torch import nn
import torch.nn.functional as F


class DiceLossOld(nn.Module):
    """from https://www.jeremyjordan.me/semantic-segmentation/"""

    def __init__(self):
        super().__init__()
        self.epsilon = 1e-6

    def forward(self, y_pred, y):
        y_pred = torch.sigmoid(y_pred)
        numerator = 2. * (y_pred * y).sum() + self.epsilon
        denominator = y_pred.sum() + y.sum() + self.epsilon
        loss = 1 - (numerator / denominator)

        return loss


"""Adapted from https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch"""
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets)
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


