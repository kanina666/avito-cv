"""Модель: компактный timm-бэкбон с одним логитом (p_180 = sigmoid(logit))."""
import timm
import torch

import config as C


def build_model(pretrained: bool = True) -> torch.nn.Module:
    return timm.create_model(C.BACKBONE, pretrained=pretrained, num_classes=1)
