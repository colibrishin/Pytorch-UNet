import torch
from torch import Tensor

def iou_score(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    iou = 0
    input = torch.squeeze(input, 0).type(torch.bool)
    target = torch.squeeze(target, 0).type(torch.bool)

    union = torch.logical_or(input, target).sum()
    intersection = torch.logical_and(input, target).sum()
    iou = intersection / union

    return iou

def iou_road(input: Tensor, target: Tensor, n_class: int):
    assert input.size() == target.size()
    iou = 0
    input = torch.squeeze(input, 0).type(torch.bool)[n_class]
    target = torch.squeeze(target, 0).type(torch.bool)[n_class]

    union = torch.logical_or(input, target).sum()
    intersection = torch.logical_and(input, target).sum()
    iou = intersection / union

    return iou

def iou_sidewalk(input: Tensor, target: Tensor, n_class: int):
    assert input.size() == target.size()
    iou = 0
    input = torch.squeeze(input, 0).type(torch.bool)[n_class]
    target = torch.squeeze(target, 0).type(torch.bool)[n_class]

    union = torch.logical_or(input, target).sum()
    intersection = torch.logical_and(input, target).sum()
    iou = intersection / union

    return iou