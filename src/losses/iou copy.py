import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class IoU_nobg(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_nobg, self).__init__()

    def forward(self, pred, target, n_classes=2, smooth=1):
        ious = []
        pred = pred.view(-1)
        target = target.view(-1)

        # Ignore IoU for background class ("0")
        for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
            union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
            if union == 0:
                ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
            else:
                ious.append(float(intersection) / float(max(union, 1)))
        print(ious)
        return ious[0]
