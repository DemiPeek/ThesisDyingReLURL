# Python script by Jacob Kooi
import os
from environments.maze_env import Maze
from agents.unsupervised_agent_reward_finding_basic import Agent_Reward_Finding_Basic
from utils import strtobool, fill_buffer, set_seed, visualize_buffer_batch
import numpy as np
import time
import argparse
import wandb
# wandb.login()

# Arguments, can change them in the command line with: python main_reward_finding_basic.py --arg_name=arg_value
parser = argparse.ArgumentParser()
parser.add_argument('--run_description', type=str, default='test_fourmaze')
parser.add_argument('--GPU', type=str, default='0')
parser.add_argument('--device', type=str, default='mps')  # gpu
parser.add_argument('--iterations', type=int, default=150000)
parser.add_argument('--latent_dim', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--eps_start', type=float, default=0.1)
parser.add_argument('--lr_encoder', type=float, default=5e-5)
parser.add_argument('--lr_dqn', type=float, default=5e-5)
parser.add_argument('--format', type=str, default='png')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.85)
parser.add_argument('--tau', type=float, default=0.02)
parser.add_argument('--showplot', type=strtobool, default=False)
parser.add_argument('--interval_iterations', type=int, default=10000)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--fill_buffer', type=int, default=20000)
parser.add_argument('--map_type', type=str, default='random_with_rewards')  # path_finding or random_with_rewards' 150-200 iterations
parser.add_argument('--maze_rewards', type=int, default=3)
parser.add_argument('--maze_size', type=int, default=8)
parser.add_argument('--higher_dim_bool', type=strtobool, default=True)
parser.add_argument('--experiment', type=str, default="standard")  # standard or bounded_loss or reincarnate or layernorm
parser.add_argument('--reincarnate_type', type=str, default="standard")  # if using reincarnate, make sure to set it to lrelu_inf or lrelu_1 or lrelu_x
parser.add_argument('--relu_bound', type=float, default=0.3)
parser.add_argument('--reward_scale', type=float, default=1)
parser.add_argument('--optimizer', type=str, default="adam")
args = parser.parse_args()

os.environ["WANDB_SILENT"] = "true"
wandb.init(
    project="project_relu_demi",
    config=vars(args),
)

# Define wandb metrics to get nice plots
wandb.define_metric("global_iterations")
# define which metrics will be plotted against it
wandb.define_metric("average_reward", step_metric="global_iterations")
wandb.define_metric("Best reward", step_metric="global_iterations")
wandb.define_metric("iterations", step_metric="global_iterations")
wandb.define_metric("q_loss", step_metric="global_iterations")
wandb.define_metric("fully_dead_relu", step_metric="global_iterations")
wandb.define_metric("weighted_dead_relu", step_metric="global_iterations")
wandb.define_metric("fraction_leaky_relu", step_metric="global_iterations")
wandb.define_metric("nr_leaky_relu", step_metric="global_iterations")
wandb.define_metric("normalized_l2norm_weight_change", step_metric="global_iterations")
wandb.define_metric("fraction_weighted_dead_relu", step_metric="global_iterations")
wandb.define_metric("fraction_fully_dead_relu", step_metric="global_iterations")
wandb.define_metric("avg_magnitude_weights", step_metric="global_iterations")
wandb.define_metric("l2_norm_grad", step_metric="global_iterations")
wandb.define_metric("normalized_l2_norm_grad", step_metric="global_iterations")
wandb.define_metric("gradient_absolute_sum", step_metric="global_iterations")
wandb.define_metric('mean_weights', step_metric="global_iterations")
wandb.define_metric('std_weights', step_metric="global_iterations")
wandb.define_metric('ratio_neg_weights', step_metric="global_iterations")
wandb.define_metric('mean_layer_without_activation', step_metric="global_iterations")
wandb.define_metric('std_layer_without_activation', step_metric="global_iterations")
wandb.define_metric('ratio_layer_neg_without_activation', step_metric="global_iterations")

set_seed(seed=wandb.config.seed)
rng = np.random.RandomState(123456)

# Create the environment
env = Maze(higher_dim_obs=args.higher_dim_bool, map_type=args.map_type, maze_size=args.maze_size, n_rewards=args.maze_rewards)
eval_env = Maze(higher_dim_obs=args.higher_dim_bool, map_type=args.map_type, maze_size=args.maze_size, n_rewards=args.maze_rewards)

env.create_map()
eval_env.create_map()

# Create the agent
agent = Agent_Reward_Finding_Basic(env, eval_env, args=args)

# Pre-fill the buffer
fill_buffer(agent.buffer, args.fill_buffer, env)

# Calculate how sparse the rewards in this environment are, for visualisation purposes
number_of_succesfull_episodes = np.where(np.array(agent.buffer.rewards) == 1)[0].shape[0]
percentage_chance_of_success = (number_of_succesfull_episodes / len(agent.buffer)) * 100
print("Number of rewards collected: ", number_of_succesfull_episodes, "Initial buffer size: ", len(agent.buffer))
print("Percentage chance of positive reward: ", percentage_chance_of_success, "%")

# # Uncomment this if you want to visualize a list of samples in the buffer.
# visualize_buffer_batch(agent)

# Main training loop
start_time = time.time()
intermediate_time = None
for i in range(args.iterations + 500):
    # Train the agent for an iteration
    agent.run_agent()
    if i % 5000 == 0:
        end_time = time.time()
        print("Iteration: ", i, "Time: ", end_time - start_time)
        start_time = time.time()
