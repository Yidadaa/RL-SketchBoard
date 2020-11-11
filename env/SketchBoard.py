from __future__ import annotations
from typing import *
import time

from gym import Env, spaces
import numpy as np
import cv2
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F

from model.AppreciatorNet import AppreciatorNet

class SketchBoardEnv(Env):
    """Definition of SketchBoard."""

    def __init__(self, target_image: np.ndarray, action_dim: int = 6, max_stroke_width: int = 10):
        """
        Constructor of SketchBoard.

        Args:
            target_image: the target image of the drawing task
            action_dim: the number of actions: color, width, start_x, start_y, end_x, end_y

            TODO:
            1. we have to determine the color mix mode (y or n).
        """
        assert isinstance(target_image, np.ndarray), \
            'target image should be numpy.ndarray, but is: {}'.format(type(target_image))
        self.target_image = target_image
        self.target_image_tensor = torch.Tensor(target_image).unsqueeze().unsqueeze()
        self.height, self.width = self.target_image.shape
        self.action_space = spaces.Box(low=np.zeros(action_dim, np.float), \
            high=np.ones(action_dim, np.float), dtype=np.float)
        self.observation_space = None
        self.reward_range = None
        self.max_stroke_width = max_stroke_width
        self.last_time = time.time()
        self.encoder = AppreciatorNet(self.width, self.height, 1)
        # init environment
        self._prepare_env()

    def _prepare_env(self):
        self.sketch_board: Image.Image = Image.fromarray(
            np.ones((self.height, self.width), np.uint8) * 255)

    def step(self, action: np.array) -> Tuple[np.array, float, bool, dict]:
        """
        Take an action to the env.

        Args:
            action: the action vector

        Return:
            observation: observation vector
            reward: total reward
            done: game is done
            info: other information
        """
        color, width, sx, sy, ex, ey = action
        start_posx, start_posy = int(sx * self.width), int(sy * self.height)
        end_posx, end_posy = int(ex * self.width), int(ey * self.height)
        int_color = int(color * 255)
        stroke_width = int(self.max_stroke_width * width)
        ImageDraw.Draw(self.sketch_board).line(xy=[(start_posx, start_posy), (end_posx, end_posy)], fill=int_color, width=stroke_width)
        return self.get_observation(), self.get_reward(), False, {}

    def render(self) -> Image:
        """Render and return sketch board content."""
        return self.sketch_board

    def close(self):
        """Close environment and clean up."""
        raise NotImplementedError

    def reset(self):
        """Reset whole env."""
        self._prepare_env()

    def get_observation(self, with_tensor=True) -> torch.Tensor:
        """Get observation array."""
        return self.encoder(self.get_buffer(with_tensor))

    def get_buffer(self, with_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """Get image buffer."""
        return (torch.Tensor if with_tensor else np.array)(self.sketch_board, np.uint8)

    def get_reward(self) -> float:
        """Get reward."""
        target_feat = self.encoder(self.target_image_tensor)
        obs_feat = self.encoder(self.get_observation())
        return F.l1_loss(obs_feat, target_feat).sum()

    def show(self):
        """Display sketchboard."""
        frame_buffer = self.get_buffer()
        # self.sketch_board.show()
        cv2.imshow("SketchBoard", frame_buffer)
        cv2.waitKey(1)

        fps = 1 / (time.time() - self.last_time)
        print(fps)
        self.last_time = time.time()