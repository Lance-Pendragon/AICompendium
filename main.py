from tianshou.env import PettingZooEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import MultiAgentPolicyManager, DQNPolicy
from tianshou.trainer import OnpolicyTrainer

# Import your custom environment
from custom_environments.tic_tac_toe.env.tic_tac_toe_env import raw_env  # Replace with your actual module

# Wrap your PettingZoo environment
env = PettingZooEnv(raw_env())

# Define policies for each agent
# need to add model, optimizer, and action space for these policies
policies = MultiAgentPolicyManager([DQNPolicy(), DQNPolicy()], env)

# Create a collector and replay buffer
collector = Collector(policies, env, VectorReplayBuffer(10000))

# Train the agents
trainer = OnpolicyTrainer(
    policies,
    env,
    collector,
    max_epoch=10,   
    step_per_epoch=1000,
)

# Run training
trainer.run()