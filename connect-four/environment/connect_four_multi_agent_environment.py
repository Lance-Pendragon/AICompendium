import gym
from gym import spaces
import numpy as np
from random import randint
from ray.rllib.env.multi_agent_env import MultiAgentEnv

NUM_ROWS = 6
NUM_COLUMNS = 7


class ConnectFourMultiAgentEnv(MultiAgentEnv):

    def __init__(self, config=None):
        super().__init__()

        # -1 = empty, 0 = player 1, 1 = player 2
        self.board = np.full((NUM_ROWS, NUM_COLUMNS), -1, dtype=np.int8)

        self.agents = self.possible_agents = ["player_1", "player_2"]
        self.currentAgentIndex = 0

        self.action_space = spaces.Discrete(NUM_COLUMNS)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=-1, high=1, shape=(NUM_ROWS, NUM_COLUMNS)
                ),  # connect4 has 6 rows, 7 columns
                "action_mask": spaces.MultiBinary(NUM_COLUMNS),
            }
        )

    def reset(self):
        # 0 = empty, 1 = player 1, 2 = player 2
        self.board = np.full((NUM_ROWS, NUM_COLUMNS), -1, dtype=np.int8)
        self.currentAgentIndex = randint(0, 1)
        return self.get_observation_space()

    def step(self, action_dict):
        current_player = self.agents[self.currentAgentIndex]
        opponent = self.agents[1 - self.currentAgentIndex]
        column = action_dict[current_player]
        rewards = {current_player: 0.0}
        terminateds = {"__all__": False}
        

        # action should be an integer corresponding with a column
        if not self.is_valid_move(column):
            reward = -10
            done = True
            return self.get_observation_space(), reward, done, {}

        row, column = self.applyMove(column)

        if self.is_victory(row, column):
            rewards[current_player] += 10
            rewards[opponent] -= 10
            terminateds = {"__all__": True}
        elif self.is_draw():
            rewards[current_player] -= 5
            rewards[opponent] -= 5
            terminateds = {"__all__": True}
        else:
            reward = 0.1
            done = False

        self.currentAgentIndex = 1 - self.currentAgentIndex

        return self.get_observation_space(), reward, done, {}

    def render(self):
        return None

    def get_observation_space(self):
        return {
            "board": np.array(self.board, dtype=np.int8),
            "action_mask": self.get_action_mask(),
        }

    def get_action_mask(self):
        return [int(not self.board[0][column] != -1) for column in range(NUM_COLUMNS)]

    def is_valid_move(self, column):
        # only need to check if top is filled
        return self.board[0][column] != -1

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
        return np.all(self.board != -1)

    def applyMove(self, column):
        for row in reversed(range(NUM_ROWS)):
            isCellEmpty = self.board[row][column] == -1
            if isCellEmpty:
                self.board[row][column] = self.currentAgentIndex
                return row, column
