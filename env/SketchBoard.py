class SketchBoardEnv(object):
    def __init__(self, width: int, height: int, action_dim: int):
        '''
        Constructor of SketchBoard.
        '''
        self.action_space = None
        self.observation_space = None
        self.reward_range = None
        self.sketch_board = None # np.array

    def step(action: list) -> tuple:
        '''
        Take an action to the env.
        '''
        raise NotImplementedError

    def render():
        '''
        Render and return sketch board content.
        '''
        return self.sketch_board

    def close():
        '''
        Close environment and clean up.
        '''
        raise NotImplementedError

    def seed():
        raise NotImplementedError