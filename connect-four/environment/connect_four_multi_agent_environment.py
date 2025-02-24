import gym
from gym import spaces
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

NUM_ROWS = 6
NUM_COLUMNS = 7


class ConnectFourMultiAgentEnv(MultiAgentEnv):

    def __init__(self, config=None):
        super().__init__()

        # 0 = empty, 1 = player 1, 2 = player 2
        self.board = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=np.int8)

        self.players = ["player_1", "player_2"]
        self.currentPlayerIndex = 0

        self.action_space = spaces.Discrete(NUM_COLUMNS)
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
        return self.get_observation_space()

    def step(self, action):
        # action should be an integer corresponding with a column
        if not self.is_valid_move(column):
            reward = -10
            done = True
            return self.get_observation_space(), reward, done, {}

        row, colunm = self.applyMove(action)

        if self.is_victory(row, column):
            reward = 100
            done = True
        elif self.is_draw():
            reward = 0
            done = True
        else:
            reward = 0.1
            done = False

        return self.get_observation_space(), reward, done, {}

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
        return [int(not self.board[0][column] != 0) for column in range(NUM_COLUMNS)]

    def is_valid_move(self, column):
        # only need to check if top is filled
        return self.board[NUM_ROWS - 1][column] != 0

    def is_victory(self, row, column):
        directions = [
            (0, 1),  # Horizontal to the right
            (1, 0),  # Vertical upwards
            (1, 1),  # Diagonal, down + right
            (1, -1),  # Diagonal, down + left
        ]

        currentPlayerPiece = self.board[row][column]

        for horizontalDirection, verticalDirection in directions:
            count = 1
            currentRow = row + horizontalDirection
            currentColumn = column + verticalDirection
            while (
                0 <= currentRow < NUM_ROWS
                and 0 <= currentColumn < NUM_COLUMNS
                and self.board[currentRow][currentColumn] == currentPlayerPiece
            ):
                count += 1
                currentRow += horizontalDirection
                currentColumn += verticalDirection

            currentRow = row - horizontalDirection
            currentColumn = column - verticalDirection
            while (
                0 <= currentRow < NUM_ROWS
                and 0 <= currentColumn < NUM_COLUMNS
                and self.board[currentRow][currentColumn] == currentPlayerPiece
            ):
                count += 1
                currentRow -= horizontalDirection
                currentColumn -= verticalDirection

            if count >= 4:
                return True
        return False

    def is_draw(self):
        return np.all(self.board != 0)

    def applyMove(self, column):
        for row in range(NUM_ROWS):
            isCellEmpty = self.board[row][column] == 0
            if isCellEmpty:
                self.board[row][column] = self.currentPlayerIndex + 1
                return row, column
