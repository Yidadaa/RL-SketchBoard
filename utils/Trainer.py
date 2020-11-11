from __future__ import annotations
from typing import *

import tianshou as ts
from tianshou import trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from env.SketchBoard import SketchBoardEnv

class Trainer:
  """Trainer of sketch board with tianshou."""

  def __init__(self):
    """ Define trainer.

    Args:
      TODO: a ba a ba
    """
    self.dummy_env = SketchBoardEnv()
    self.training_target_images: List[np.ndarray] = []
    self.test_target_images: List[np.ndarray] = []

  def make_envs(self, env_count: int, target_imgs: List[np.ndarray]) -> DummyVectorEnv:
    """Create a group of environments.

    Args:
      env_count: number of envs

    Returns:
      envs: vector of envs
    """
    return ts.env.DummyVectorEnv(
      [lambda: SketchBoardEnv(target_imgs[i % len(target_imgs)])
        for i in range(env_count)])
