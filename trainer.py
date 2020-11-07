from env.SketchBoard import SketchBoardEnv
import numpy as np
from time import sleep

def test():
  skb = SketchBoardEnv(400, 400)
  test_action = np.array([0.5, 0.2, 0.1, 0.1, 0.2, 0.2])
  for i in range(100):
    random_action = skb.action_space.sample()
    skb.step(random_action)
    skb.show()
    sleep(0.03)

test()