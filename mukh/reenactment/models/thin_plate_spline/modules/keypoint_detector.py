"""
Source: https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model

LICENSE: MIT
"""

import torch
from torch import nn
from torchvision import models


class KPDetector(nn.Module):
    """
    Predict K*5 keypoints.
    """

    def __init__(self, num_tps, **kwargs):
        super(KPDetector, self).__init__()
        self.num_tps = num_tps

        self.fg_encoder = models.resnet18(pretrained=False)
        num_features = self.fg_encoder.fc.in_features
        self.fg_encoder.fc = nn.Linear(num_features, num_tps * 5 * 2)

    def forward(self, image):

        fg_kp = self.fg_encoder(image)
        (
            bs,
            _,
        ) = fg_kp.shape
        fg_kp = torch.sigmoid(fg_kp)
        fg_kp = fg_kp * 2 - 1
        out = {"fg_kp": fg_kp.view(bs, self.num_tps * 5, -1)}

        return out
