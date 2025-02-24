import gym
from gym import spaces
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

NUM_ROWS = 6
NUM_COLUMNS = 7

class ConnectFourMultiAgentEnvironment(MultiAgentEnv):

    def __init__(self, config=None):
        super().__init__()

        # 0 = empty, 1 = player 1, 2 = player 2
        self.board = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=np.int8)

        self.action_space = spaces.Discrete(7)  # 7 columns to pick from
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0, high=2, shape=(NUM_ROWS, NUM_COLUMNS)
                ),  # connect4 has 6 rows, 7 columns
                "action_mask": spaces.MultiBinary(NUM_COLUMNS),
            }
        )

    def reset(self):        
        # 0 = empty, 1 = player 1, 2 = player 2
        self.board = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=np.int8)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0, high=2, shape=(NUM_ROWS, NUM_COLUMNS)
                ),  # connect4 has 6 rows, 7 columns
                "action_mask": spaces.MultiBinary(NUM_COLUMNS),
            }
        )
        return None

    def step(self, action):
        # action should be an integer corresponding with a column
        if not self.is_valid_move():
           reward = -10
           done = True
        elif self.is_victory():
            reward = 100
            done = True
        elif self.is_draw():
            reward = 0
            done = True
        else:
            reward = -.1
            done = False

        return self.get_observation_space, reward, done, {}

    def render(self):
        return None

    def get_observation_space(self):
        return spaces.Dict(
            {
                "board": np.array(self.board, dtype=np.int8),
                "action_mask": self.get_action_mask(),
            }
        )

    def get_action_mask(self):
        return [int(not self.board[0][col] != 0) for col in range(NUM_COLUMNS)]


    def is_valid_move(self, action):
        return self.board[action] == '0'