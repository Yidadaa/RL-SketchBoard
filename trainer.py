from env.SketchBoard import SketchBoardEnv

def test():
  skb = SketchBoardEnv(300, 400, 10)
  skb.show()

test()