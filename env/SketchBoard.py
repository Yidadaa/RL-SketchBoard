import numpy as np
import cv2
from tkinter import Canvas, Frame

from PIL import Image, ImageDraw

class TKEnv(Frame):
    def __init__(self, width: int, height: int) -> None:
        Frame.__init__(self)
        self.frame = Frame(self)
        self.canvas = Canvas(self.frame, width=width, height=height, bg='white')

    def draw(self, line):
        return True

    def get_pixels(self):
        pass

class SketchBoardEnv(object):
    """Definition of SketchBoard."""

    def __init__(self, width: int, height: int, action_dim: int = 6):
        """
        Constructor of SketchBoard.

        Args:
            width: the width of sketch board
            height: the height of sketch board
            action_dim: the number of actions: color, width, start_x, start_y, end_x, end_y

            TODO:
            1. we have to determine the color mix mode (y or n).
            2. we should replace tkenv with PIL.Image and PIL.ImageDraw
        """
        self.action_space = None
        self.observation_space = None
        self.reward_range = None
        self.sketch_board = np.ones((width, height), np.float)  # Image
        self.tkenv = TKEnv(width, height)

    def _perpare_env(self):
        """Prepare variables for sketch board."""
        pass

    def step(self, action: np.array) -> tuple:
        """
        Take an action to the env.

        Args:
            action: the action vector
        """
        color, width, sx, sy, ex, ey = action
        self.canvas

        raise NotImplementedError

    def render(self):
        """Render and return sketch board content."""
        return self.sketch_board

    def close(self):
        """Close environment and clean up."""
        raise NotImplementedError

    def seed(self):
        """Generate seed for env."""
        raise NotImplementedError

    def show(self):
        """Display sketchboard."""
        cv2.imshow("SketchBoard", self.sketch_board)
        cv2.waitKey(0)