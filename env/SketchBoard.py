from gym import Env, spaces
import numpy as np
import cv2
from PIL import Image, ImageDraw

import time

class SketchBoardEnv(Env):
    """Definition of SketchBoard."""

    def __init__(self, width: int, height: int, action_dim: int = 6, max_stroke_width: int = 10):
        """
        Constructor of SketchBoard.

        Args:
            width: the width of sketch board
            height: the height of sketch board
            action_dim: the number of actions: color, width, start_x, start_y, end_x, end_y

            TODO:
            1. we have to determine the color mix mode (y or n).
            2. we should replace tkenv with PIL.Image and PIL.ImageDraw. [done]
        """
        self.height, self.width = height, width
        self.action_space = spaces.Box(low=np.zeros(action_dim, np.float), high=np.ones(action_dim, np.float), dtype=np.float)
        self.observation_space = None
        self.reward_range = None
        self.max_stroke_width = max_stroke_width
        self._prepare_env()
        self.last_time = time.time()

    def _prepare_env(self):
        self.sketch_board = Image.fromarray(np.ones((self.height, self.width), np.uint8) * 255)  # Image

    def step(self, action: np.array) -> tuple:
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

    def render(self) -> Image:
        """Render and return sketch board content."""
        return self.sketch_board

    def close(self):
        """Close environment and clean up."""
        raise NotImplementedError

    def reset(self):
        """Reset whole env."""
        self._prepare_env()

    def show(self):
        """Display sketchboard."""
        frame_buffer = np.array(self.sketch_board, np.uint8)
        # self.sketch_board.show()
        cv2.imshow("SketchBoard", frame_buffer)
        cv2.waitKey(1)

        fps = 1 / (time.time() - self.last_time)
        print(fps)
        self.last_time = time.time()