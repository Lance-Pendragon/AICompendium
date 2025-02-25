import functools
import numpy as np
from gymnasium.spaces import Discrete, Box, Dict
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide variety of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "tic_tac_toe_v0"}

    def __init__(self, render_mode=None):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.agents = self.possible_agents = ["X", "O"]
        self.grid = np.full((3, 3), -1, dtype=np.int8)  # -1: empty, 0: X, 1: O

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self.render_mode = render_mode

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Dict(
            {
                "observation": Box(low=-1, high=1, shape=(3, 3), dtype=np.int8),
                "action_mask": Box(low=0, high=1, shape=(9,), dtype=np.int8),
            }
        )

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(9)  # 9 spots in tic tac toe

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        # Print the grid in a human-readable format
        symbols = {-1: ".", 0: "X", 1: "O"}
        for row in self.grid:
            print(" ".join(symbols[cell] for cell in row))
        print()

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # Create the action mask
        action_mask = np.zeros(9, dtype=np.int8)
        for i in range(9):
            row, col = divmod(i, 3)
            if self.grid[row, col] == -1:  # Cell is empty
                action_mask[i] = 1

        return {"observation": np.array(self.grid, dtype=np.int8), "action_mask": action_mask}

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        """
        self.grid = np.full((3, 3), -1, dtype=np.int8)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        return self.observe(self.agent_selection)

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        agent = self.agent_selection
        opponent = self._agent_selector.next()

        # Check if the action is valid using the action mask
        observation = self.observe(agent)
        if observation["action_mask"][action] == 0:
            raise ValueError(f"Invalid action {action}: cell already occupied")

        # Update the grid
        row, col = divmod(action, 3)
        self.grid[row, col] = 0 if agent == "X" else 1

        # Check for a win
        if self._check_win(agent):
            self.rewards[agent] = 1
            self.rewards[opponent] = -1
            self.terminations = {agent: True for agent in self.agents}
        # Check for a draw
        elif np.all(self.grid != -1):
            self.rewards = {agent: 0 for agent in self.agents}
            self.terminations = {agent: True for agent in self.agents}
        else:
            self.rewards = {agent: 0 for agent in self.agents}

        # Switch to the next agent
        self.agent_selection = opponent

        # Accumulate rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def _check_win(self, agent):
        """
        Helper method to check if the current agent has won.
        """
        value = 0 if agent == "X" else 1

        # Check rows and columns
        for i in range(3):
            if np.all(self.grid[i, :] == value) or np.all(self.grid[:, i] == value):
                return True

        # Check diagonals
        if np.all(np.diag(self.grid) == value) or np.all(np.diag(np.fliplr(self.grid)) == value):
            return True

        return False

        
env = env(render_mode="human")
env.reset()

for _ in range(9):  # Maximum of 9 moves in Tic-Tac-Toe
    agent = env.agent_selection
    observation = env.observe(agent)
    valid_actions = np.where(observation["action_mask"] == 1)[0]
    if len(valid_actions) == 0:
        print("No valid actions left!")
        break
    action = np.random.choice(valid_actions)  # Choose a random valid action
    env.step(action)
    if any(env.terminations.values()):
        print(f"Game over! Rewards: {env.rewards}")
        break