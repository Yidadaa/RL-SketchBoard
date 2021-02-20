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
from PIL import Image

from env.SketchBoard import SketchBoardEnv
from model.AppreciatorNet import AppreciatorNet


class Trainer:
    """Trainer of sketch board with tianshou."""

    def __init__(self, w: int, h: int, train_target_imgs: List[str], test_target_imgs: List[str],
                 train_env_count: int = 8, test_env_count: int = 100) -> None:
        """ Define trainer.

        Args:
          train_env_count: number of training envs
          test_env_count: number of testing envs
        """
        self.w = w, self.h = h
        self.dummy_env = SketchBoardEnv()

        self.train_env_count = train_env_count
        self.test_env_count = test_env_count

        self.training_target_images = self.load_images(train_target_imgs)
        self.test_target_images = self.load_images(test_target_imgs)

        self.train_envs = self.make_envs(
            train_env_count, self.training_target_images)
        self.test_envs = self.make_envs(
            test_env_count, self.test_target_images)
        self.net = None # TODO: build net according to pong_ppo.py

    def load_images(image_paths: List[str]) -> List[np.ndarray]:
        """Load images."""
        return [np.array(Image.open(fp), dtype=np.uint8) for fp in image_paths]

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

    def build_net(self, ) -> AppreciatorNet:
        """Build action net."""
        return AppreciatorNet()
