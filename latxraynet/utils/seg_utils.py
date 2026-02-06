
import torch


#-------------------------------------------------------------------------
def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-5
    intersection = (pred * target).sum()
    return 1 - ((2 * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))

#-------------------------------------------------------------------------
def dice_coeff(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    smooth = 1e-5
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

#-------------------------------------------------------------------------
def iou_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / (union + 1e-5)