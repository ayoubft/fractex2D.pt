import torch
import torch.nn.functional as F


def labels_to_one_hot(labels, num_classes):
    return F.one_hot(labels.long(), num_classes=num_classes).float()


def generalised_dice_loss(prediction, ground_truth, weight_map=None, type_weight='Square'):
    prediction = prediction.float()
    
    if ground_truth.shape == prediction.shape:
        ground_truth = ground_truth[..., -1]

    one_hot = labels_to_one_hot(ground_truth, prediction.shape[1])
    one_hot = one_hot.permute(0, 4, 1, 2, 3) if one_hot.ndim == 5 else one_hot.permute(0, 3, 1, 2)

    prediction = prediction.view(prediction.shape[0], prediction.shape[1], -1)
    one_hot = one_hot.view(one_hot.shape[0], one_hot.shape[1], -1)

    if weight_map is not None:
        weight_map = weight_map.view(weight_map.shape[0], -1, 1)
        weight_map_nclasses = weight_map.expand(-1, -1, prediction.shape[1])
        
        ref_vol = torch.sum(weight_map_nclasses * one_hot.transpose(1, 2), dim=1)
        intersect = torch.sum(weight_map_nclasses * one_hot.transpose(1, 2) * prediction.transpose(1, 2), dim=1)
        seg_vol = torch.sum(weight_map_nclasses * prediction.transpose(1, 2), dim=1)
    else:
        ref_vol = torch.sum(one_hot, dim=2)
        intersect = torch.sum(one_hot * prediction, dim=2)
        seg_vol = torch.sum(prediction, dim=2)

    if type_weight == 'Square':
        weights = 1.0 / (ref_vol ** 2 + 1e-8)
    elif type_weight == 'Simple':
        weights = 1.0 / (ref_vol + 1e-8)
    elif type_weight == 'Uniform':
        weights = torch.ones_like(ref_vol)
    else:
        raise ValueError(f"type_weight \"{type_weight}\" is not defined.")

    weights[torch.isinf(weights)] = 0
    if torch.any(weights == 0):
        weights[weights == 0] = torch.max(weights)

    numerator = 2 * torch.sum(weights * intersect, dim=1)
    denominator = torch.sum(weights * torch.clamp(seg_vol + ref_vol, min=1.0), dim=1)

    dice_score = numerator / denominator
    dice_score[torch.isnan(dice_score)] = 1.0

    return 1 - dice_score.mean()
