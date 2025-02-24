import gym
from gym import spaces
import numpy as np
from random import randint, choice
from ray.rllib.env.multi_agent_env import MultiAgentEnv

NUM_ROWS = 6
NUM_COLUMNS = 7


class ConnectFourMultiAgentEnv(MultiAgentEnv):

    def __init__(self, config=None):
        super().__init__()

        self.board = None
        self.agents = self.possible_agents = ["player_1", "player_2"]
        self.current_agent = None
        
        self.action_spaces = {
            agent: spaces.Discrete(NUM_COLUMNS) for agents in self.agents
        }
        self.observation_spaces = {
            agent: self.get_observation() for agents in self.agents
        }

    def reset(self):
        self.board = np.full((NUM_ROWS, NUM_COLUMNS), -1, dtype=np.int8)
        self.current_agent = random.choice(self.agents)

        return {
            self.current_agent: self.get_observation()
        }

    def step(self, action_dict):
        column = self.current_agent
        opponent = "player_1" if self.current_agent == "player_1" else "player_2"

        column = action_dict[self.current_agent]

        rewards = {self.current_agent: 0.0}
        terminateds = {"__all__": False}

        if not self.is_valid_move(column):
            rewards[current_player] -= 5

        row, column = self.applyMove(column)

        if self.is_victory(row, column):
            rewards[self.current_agent] = 10
            rewards[opponent] = -10
            terminateds["__all__"] = True
        elif self.is_draw():
            rewards[self.current_agent] = -5
            rewards[opponent] = -5
            terminateds["__all__"] = True

        # todo - find a way to 'grade' non game ending moves
        # promote connections?

        self.current_agent = opponent

        return (
            {self.current_agent: self.get_observation},
            rewards,
            terminateds,
            {},
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
        return np.all(self.board != -1)
