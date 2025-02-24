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
        self.agents = ["player_1", "player_2"]
        self.current_agent_index = randint(0, 1)

        # Shared action and observation spaces
        self.action_space = spaces.Discrete(NUM_COLUMNS)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=-1, high=1, shape=(NUM_ROWS, NUM_COLUMNS), dtype=np.int8
                ),
                "action_mask": spaces.MultiBinary(NUM_COLUMNS),
            }
        )

    def reset(self):
        self.board = np.full((NUM_ROWS, NUM_COLUMNS), -1, dtype=np.int8)
        self.current_agent_index = randint(0, 1)

        return {agent: self.get_observation() for agent in self.agents}

    def step(self, action_dict):
        current_player = self.agents[self.current_agent_index]
        opponent = self.agents[1 - self.current_agent_index]
        column = action_dict[current_player]

        rewards = {current_player: 0.0, opponent: 0.0}
        terminateds = {current_player: False, opponent: False, "__all__": False}

        if not self.is_valid_move(column):
            rewards[current_player] = -10  # Penalty for invalid move
            terminateds["__all__"] = True
            return (
                {agent: self.get_observation() for agent in self.agents},
                rewards,
                terminateds,
                {},
            )

        row, column = self.applyMove(column)

        if self.is_victory(row, column):
            rewards[current_player] = 10
            rewards[opponent] = -10
            terminateds["__all__"] = True
        elif self.is_draw():
            rewards[current_player] = -5
            rewards[opponent] = -5
            terminateds["__all__"] = True

        self.current_agent_index = 1 - self.current_agent_index

        return (
            {agent: self.get_observation() for agent in self.agents},
            rewards,
            terminateds,
            {},
        )

    def get_observation(self):
        return {
            "board": np.array(self.board, dtype=np.int8),
            "action_mask": self.get_action_mask(),
        }

    def get_action_mask(self):
        return [1 if self.board[0][col] == -1 else 0 for col in range(NUM_COLUMNS)]

    def is_valid_move(self, column):
        return self.board[0][column] == -1

    def applyMove(self, column):
        for row in reversed(range(NUM_ROWS)):
            if self.board[row][column] == -1:
                self.board[row][column] = self.current_agent_index
                return row, column

    def is_victory(self, row, column):
        directions = [
            (0, 1),  # Horizontal to the right
            (1, 0),  # Vertical upwards
            (1, 1),  # Diagonal, down + right
            (1, -1),  # Diagonal, down + left
        ]

        current_player_piece = self.board[row][column]

        for horizontal_direction, vertical_direction in directions:
            count = 1
            currentRow = row + horizontal_direction
            currentColumn = column + vertical_direction
            while (
                0 <= currentRow < NUM_ROWS
                and 0 <= currentColumn < NUM_COLUMNS
                and self.board[currentRow][currentColumn] == current_player_piece
            ):
                count += 1
                currentRow += horizontal_direction
                currentColumn += vertical_direction

            currentRow = row - horizontal_direction
            currentColumn = column - vertical_direction
            while (
                0 <= currentRow < NUM_ROWS
                and 0 <= currentColumn < NUM_COLUMNS
                and self.board[currentRow][currentColumn] == current_player_piece
            ):
                count += 1
                currentRow -= horizontal_direction
                currentColumn -= vertical_direction

            if count >= 4:
                return True
        return False

    def is_draw(self):
        """Checks if the board is full (draw)."""
        return np.all(self.board != -1)
