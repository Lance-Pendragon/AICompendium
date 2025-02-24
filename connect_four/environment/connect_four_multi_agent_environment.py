import gym
import pygame
import numpy as np
from gym import spaces
from random import randint, choice
from ray.rllib.env.multi_agent_env import MultiAgentEnv

NUM_ROWS = 6
NUM_COLUMNS = 7


class ConnectFourMultiAgentEnv(MultiAgentEnv):

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}

        self.board = np.full((NUM_ROWS, NUM_COLUMNS), -1, dtype=np.int8)
        self.agents = self.possible_agents = ["player_1", "player_2"]
        self.current_agent = None
        
        self.action_spaces = {
            agent: spaces.Discrete(NUM_COLUMNS) for agent in self.agents
        }
        self.observation_spaces = spaces.Dict({
            agent: spaces.Dict({
                "board": spaces.Box(low=-1, high=1, shape=(NUM_ROWS, NUM_COLUMNS), dtype=np.int8),
                "action_mask": spaces.MultiBinary(NUM_COLUMNS)
            }) for agent in self.agents
        })

        # New rendering configuration
        self.should_render = self.config.get("should_render", False)
        self.render_enabled = False
        
        # Initialize rendering if requested
        if self.should_render:
            # Pygame visualization parameters
            self.cell_size = 80
            self.radius = int(self.cell_size / 2 - 5)
            self.colors = {
                -1: (255, 255, 255),  # Empty cell
                0: (255, 0, 0),        # Player 1 (Red)
                1: (255, 255, 0)       # Player 2 (Yellow)
            }
            self._init_render()

    
    def _init_render(self):
        pygame.init()
        width = NUM_COLUMNS * self.cell_size
        height = (NUM_ROWS + 1) * self.cell_size
        self.screen = pygame.display.set_mode((width, height))
        self.render_enabled = True
        self._draw_board()

    def _draw_board(self):
        if not self.render_enabled:
            return

        # Clear screen with blue background
        self.screen.fill((0, 0, 255))
        
        # Draw all game pieces
        for row in range(NUM_ROWS):
            for col in range(NUM_COLUMNS):
                pygame.draw.circle(
                    self.screen,
                    self.colors[self.board[row][col]],
                    (
                        col * self.cell_size + self.cell_size // 2,
                        (row + 1) * self.cell_size + self.cell_size // 2
                    ),
                    self.radius
                )
        
        # Draw column numbers
        font = pygame.font.Font(None, 36)
        for col in range(NUM_COLUMNS):
            text = font.render(str(col + 1), True, (255, 255, 255))
            text_rect = text.get_rect(
                center=(col * self.cell_size + self.cell_size // 2, 
                        self.cell_size // 2)
            )
            self.screen.blit(text, text_rect)
        
        pygame.display.update()

    def render(self):
        if not self.render_enabled:
            return

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        self._draw_board()
        

    def close(self):
        if self.render_enabled:
            pygame.quit()
            self.render_enabled = False

    def reset(self):
        self.board = np.full((NUM_ROWS, NUM_COLUMNS), -1, dtype=np.int8)
        self.current_agent = np.random.choice(self.agents)

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
            rewards[self.current_agent] -= 5

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
            {self.current_agent: self.get_observation()},
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
                self.board[row][column] = 0 if self.current_agent == "0" else 1
                if self.render_enabled:
                    self.render()  # Update display after move
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
