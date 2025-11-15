"""This code is borrowed from https://github.com/YilmazKadir/
Segmentation_Losses/blob/main/losses/dice.py
"""

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import softmax
from losses.utils import flatten
from typing import List


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super(DiceLoss, self).__init__()
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
            dice = (2.0 * intersection + self.smooth) / (
                input.sum() + target.sum() + self.smooth
            )
            losses.append(1 - dice)
        losses_tensor = torch.stack(losses)
        return losses_tensor.mean()
