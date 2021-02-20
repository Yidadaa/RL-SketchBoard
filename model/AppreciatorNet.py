from __future__ import annotations
from typing import *

import torch
from torch import nn
from torchvision.models import resnet18
import numpy as np

class AppreciatorNet(nn.Module):
  """Defination of network to extract features from sketch."""

  def __init__(self, w: int, h: int, c:int = 3, feat_dim: int = 64, use_resnet: bool = False) -> None:
    """Define network with width, height and channel of input image.

    Args:
      w: width of input sketch image
      h: height of input sketch image
      c: channel of input sketch image,
      feat_dim: output feature dimensions
      use_resnet: use resnet or not
    """
    super().__init__()
    self.w, self.h, self.c = w, h, c
    self.feat_dim = feat_dim
    self.use_resnet = use_resnet
    self.model = self.build_model()

  def build_model(self) -> nn.Module:
    """Build model."""
    model = None
    if self.use_resnet:
      model = nn.Sequential(
        list(resnet18().modules())[:-1],
        nn.Linear(in_features=512, out_features=self.feat_dim)
      )
    else:
      model = nn.Sequential(
        nn.Conv2d(3, 64, 7, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, 4, 1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
      )
    return model

  def forward(self, sketch_img: torch.Tensor) -> torch.Tensor:
    """Extract features from sketch_img."""
    return self.model(sketch_img)


class ActionDecoder(nn.Module):
  """Defination of action decoder."""

  def __init__(self, obs_shape: Union[Tuple[int, ...]], action_shape: Union[Tuple[int, ...]]) -> None:
    """Define action decoder.

    Args:
      obs_shape: a tuple denotes observation shape
      action_shape: a tuple denotes action shape
    """
    super().__init__()
    self.model = nn.Sequential(
      nn.Linear(np.prod(obs_shape, dtype=np.int), 128), nn.ReLU(inplace=True),
      nn.Linear(128, 128), nn.ReLU(inplace=True),
      nn.Linear(128, 128), nn.ReLU(inplace=True),
      nn.Linear(128, np.prod(action_shape, dtype=np.int))
    )

  def forward(self, x: torch.Tensor, state: Any = None, info: dict = {}) -> torch.Tensor:
    """Decode action from feature map."""
    return self.model(x), state