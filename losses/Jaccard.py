import torch
import torch.nn as nn
from torch import Tensor
from losses.utils import flatten
from torch.nn.functional import softmax
from typing import List


class JaccardLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super(JaccardLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input = softmax(input, dim=1)
        num_classes = input.size(1)
        losses: List[Tensor] = []
        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]
            intersection = (input_c * target_c).sum()
            total = (input_c + target_c).sum()
            union = total - intersection
            IoU = (intersection + self.smooth) / (union + self.smooth)
            losses.append(1 - IoU)
        losses_tensor = torch.stack(losses)
        return losses_tensor.mean()
