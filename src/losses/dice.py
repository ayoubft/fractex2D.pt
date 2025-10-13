import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        bce_weight = 0.5
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() +
                                                    targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        loss_final = BCE * bce_weight + dice_loss * (1 - bce_weight)
        return loss_final


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() +
                                                    targets.sum() + smooth)
        return dice_loss


class DiceBCELoss_with_logits(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss_with_logits, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        bce_weight = 0.5
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() +
                                                    targets.sum() + smooth)

        pos_weight = (targets == 0).sum() / (targets == 1).sum()

        # BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight)
        loss_final = BCE * bce_weight + dice_loss * (1 - bce_weight)
        return loss_final


class DiceScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceScore, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_score = (2.*intersection + smooth)/(inputs.sum() + targets.sum() +
                                                 smooth)
        return dice_score
