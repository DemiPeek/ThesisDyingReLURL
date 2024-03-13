# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "project_relu_demi"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000  # 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    experiment: str = "standard"
    """the specific experiment, can be standars/none, boundedloss_beforerelu, boundedloss_afterrelu or layernorm"""


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, experiment):
        super().__init__()
        self.experiment = experiment

        layers = [
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
        ]

        if self.experiment == "layernorm":
            layers.append(nn.LayerNorm(512))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(512, env.single_action_space.n))

        self.network = nn.Sequential(*layers)

        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.network(x / 255.0)

    def layers(self, x):
        # network[:-1] is after relu
        # in case of layernorm, [:-3] is pre-relu and [:-2] is the layernorm
        # else [:-2] is pre-relu
        return self.network[:-1](x / 255.0), self.network[:-2](x / 255.0), self.network[:-3](x / 255.0)

    def encoder_layers(self, x):
        conv1 = self.flatten(self.network[:-9](x / 255.0))
        conv2 = self.flatten(self.network[:-7](x / 255.0))
        conv3 = self.flatten(self.network[:-5](x / 255.0))
        relu1 = self.flatten(self.network[:-8](x / 255.0))
        relu2 = self.flatten(self.network[:-6](x / 255.0))
        relu3 = self.flatten(self.network[:-4](x / 255.0))
        return conv1, conv2, conv3, relu1, relu2, relu3

    def encoder_layers_layernorm(self, x):
        conv1 = self.flatten(self.network[:-10](x / 255.0))
        conv2 = self.flatten(self.network[:-8](x / 255.0))
        conv3 = self.flatten(self.network[:-6](x / 255.0))
        relu1 = self.flatten(self.network[:-9](x / 255.0))
        relu2 = self.flatten(self.network[:-7](x / 255.0))
        relu3 = self.flatten(self.network[:-5](x / 255.0))
        return conv1, conv2, conv3, relu1, relu2, relu3


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def calculate_dead_relu(layer):
    is_dead_per_data_point = (layer.abs() < 0.001)
    dead_fractions = is_dead_per_data_point.float().mean(dim=0)
    weighted_count = dead_fractions.sum()
    fully_dead = torch.sum(dead_fractions == 1.0).item()
    weighted_dead = weighted_count.item()
    frac_fully_dead = fully_dead / layer.shape[1]
    frac_weighted_dead = weighted_dead / layer.shape[1]
    return frac_fully_dead, frac_weighted_dead


def layer_metrics(layer):
    array = layer.cpu().detach().numpy()
    mean = np.mean(array)
    std = np.std(array)
    nr_neg = np.sum(array <= 0)
    ratio_neg = nr_neg / np.size(array)
    return mean, std, ratio_neg


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.define_metric("global_iterations")
        wandb.define_metric("episodic_return", step_metric="global_iterations")
        wandb.define_metric("iterations", step_metric="global_iterations")
        wandb.define_metric("q_loss", step_metric="global_iterations")
        # wandb.define_metric("fully_dead_relu", step_metric="global_iterations")
        # wandb.define_metric("weighted_dead_relu", step_metric="global_iterations")
        wandb.define_metric("fraction_fully_dead_latent", step_metric="global_iterations")
        wandb.define_metric("fraction_weighted_dead_latent", step_metric="global_iterations")
        wandb.define_metric("fraction_fully_dead_relu1", step_metric="global_iterations")
        wandb.define_metric("fraction_weighted_dead_relu1", step_metric="global_iterations")
        wandb.define_metric("fraction_fully_dead_relu2", step_metric="global_iterations")
        wandb.define_metric("fraction_weighted_dead_relu2", step_metric="global_iterations")
        wandb.define_metric("fraction_fully_dead_relu3", step_metric="global_iterations")
        wandb.define_metric("fraction_weighted_dead_relu3", step_metric="global_iterations")
        wandb.define_metric('mean_weights', step_metric="global_iterations")
        wandb.define_metric('std_weights', step_metric="global_iterations")
        wandb.define_metric('ratio_neg_weights', step_metric="global_iterations")
        wandb.define_metric('mean_preactivation', step_metric="global_iterations")
        wandb.define_metric('std_preactivation', step_metric="global_iterations")
        wandb.define_metric('ratio_neg_preactivation', step_metric="global_iterations")
        wandb.define_metric('mean_conv1', step_metric="global_iterations")
        wandb.define_metric('std_conv1', step_metric="global_iterations")
        wandb.define_metric('ratio_neg_conv1', step_metric="global_iterations")
        wandb.define_metric('mean_conv2', step_metric="global_iterations")
        wandb.define_metric('std_conv2', step_metric="global_iterations")
        wandb.define_metric('ratio_neg_conv2', step_metric="global_iterations")
        wandb.define_metric('mean_conv3', step_metric="global_iterations")
        wandb.define_metric('std_conv3', step_metric="global_iterations")
        wandb.define_metric('ratio_neg_conv3', step_metric="global_iterations")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs, args.experiment).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, args.experiment).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    wandb.log({"episodic_return": info["episode"]["r"],
                               "global_iterations": global_step
                               })

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()

                if args.experiment == "layernorm":
                    latent, _, preactivation = q_network.layers(data.observations)
                    conv1, conv2, conv3, relu1, relu2, relu3 = q_network.encoder_layers_layernorm(data.observations)
                else:
                    latent, preactivation, _ = q_network.layers(data.observations)
                    conv1, conv2, conv3, relu1, relu2, relu3 = q_network.encoder_layers(data.observations)

                loss = F.mse_loss(td_target, old_val)

                if global_step % 10000 == 0:
                    model_name = f'saved_networks/seed_{args.seed}_global_step_{global_step}.pth'
                    torch.save(q_network.state_dict(), model_name)

                    # count_dying_relu function
                    frac_fully_dead_latent, frac_weighted_dead_latent = calculate_dead_relu(latent)
                    frac_fully_dead_relu1, frac_weighted_dead_relu1 = calculate_dead_relu(relu1)
                    frac_fully_dead_relu2, frac_weighted_dead_relu2 = calculate_dead_relu(relu2)
                    frac_fully_dead_relu3, frac_weighted_dead_relu3 = calculate_dead_relu(relu3)

                    wandb.log({"fraction_fully_dead_latent": frac_fully_dead_latent,
                               "fraction_weighted_dead_latent": frac_weighted_dead_latent,
                               "fraction_fully_dead_relu1": frac_fully_dead_relu1,
                               "fraction_weighted_dead_relu1": frac_weighted_dead_relu1,
                               "fraction_fully_dead_relu2": frac_fully_dead_relu2,
                               "fraction_weighted_dead_relu2": frac_weighted_dead_relu2,
                               "fraction_fully_dead_relu3": frac_fully_dead_relu3,
                               "fraction_weighted_dead_relu3": frac_weighted_dead_relu3,
                               "global_iterations": global_step
                               })

                if args.experiment == "boundedloss_beforerelu_0.3":
                    infinity_norm = torch.norm(preactivation, float('inf'), dim=-1) - 0.3  # where 0.3 is the relu bound
                    zero = torch.zeros_like(infinity_norm)
                    max_value = torch.max(infinity_norm, zero).mean()
                    loss += max_value

                if args.experiment == "boundedloss_beforerelu_3":
                    infinity_norm = torch.norm(preactivation, float('inf'), dim=-1) - 3  # where 3 is the relu bound
                    zero = torch.zeros_like(infinity_norm)
                    max_value = torch.max(infinity_norm, zero).mean()
                    loss += max_value

                if args.experiment == "boundedloss_afterrelu_0.3":
                    infinity_norm = torch.norm(latent, float('inf'), dim=-1) - 0.3  # where 0.3 is the relu bound
                    zero = torch.zeros_like(infinity_norm)
                    max_value = torch.max(infinity_norm, zero).mean()
                    loss += max_value

                if args.experiment == "boundedloss_afterrelu_3":
                    infinity_norm = torch.norm(latent, float('inf'), dim=-1) - 3  # where 3 is the relu bound
                    zero = torch.zeros_like(infinity_norm)
                    max_value = torch.max(infinity_norm, zero).mean()
                    loss += max_value

                if global_step % 1000 == 0:
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    wandb.log({
                        "q_loss": loss,
                        "global_iterations": global_step
                    })
                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % 10000 == 0:
                    # Metrics weights of layer pre-activation
                    if args.experiment == "layernorm":
                        weights = q_network.network[-4].weight
                    else:
                        weights = q_network.network[-3].weight

                    mean_weights, std_weights, ratio_neg_weights = layer_metrics(weights)

                    mean_preactivation, std_preactivation, ratio_neg_preactivation = layer_metrics(preactivation)
                    mean_conv1, std_conv1, ratio_neg_conv1 = layer_metrics(conv1)
                    mean_conv2, std_conv2, ratio_neg_conv2 = layer_metrics(conv2)
                    mean_conv3, std_conv3, ratio_neg_conv3 = layer_metrics(conv3)

                    wandb.log({'mean_weights': mean_weights,
                               'std_weights': std_weights,
                               'ratio_neg_weights': ratio_neg_weights,
                               'mean_preactivation': mean_preactivation,
                               'std_preactivation': std_preactivation,
                               'ratio_neg_preactivation': ratio_neg_preactivation,
                               'mean_conv1': mean_conv1,
                               'std_conv1': std_conv1,
                               'ratio_neg_conv1': ratio_neg_conv1,
                               'mean_conv2': mean_conv2,
                               'std_conv2': std_conv2,
                               'ratio_neg_conv2': ratio_neg_conv2,
                               'mean_conv3': mean_conv3,
                               'std_conv3': std_conv3,
                               'ratio_neg_conv3': ratio_neg_conv3,
                               'global_iterations': global_step
                               })

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    envs.close()
