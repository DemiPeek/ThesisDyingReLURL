# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import pandas as pd
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

    num_envs: int = 1
    """the number of parallel game environments"""
    network_file: str = "saved_networks/seed_1_global_step_100000000.pth"
    """the file with the saved model"""


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        layers = [nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                  nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(),
                  nn.Linear(512, env.single_action_space.n)]

        self.network = nn.Sequential(*layers)

        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.network(x / 255.0)


def evaluate(
        model_path: str,
        eval_episode: int,
        Model: torch.nn.Module,
        device :torch.device,
        epsilon: float = 0.05,
):
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # q_network.load_state_dict(torch.load(f'saved_networks/{args.network_file}'))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episode:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            input = torch.Tensor(obs).to(device)
            noised_input = input + torch.randn_like(input)  # * 0.25
            # tensor + torch.randn(tensor.size()) * self.std + self.mean
            q_values = model(noised_input)
            # q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs
    episodic_returns = [int(arr[0]) if arr.dtype == np.int32 else float(arr[0]) for arr in episodic_returns]
    return episodic_returns


if __name__ == "__main__":
    from dqn_atari import make_env
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # device = torch.device("mps")

    saved_models = ["seed_1_global_step_53850000.pth",
                    "seed_1_global_step_60000000.pth",
                    "seed_1_global_step_70000000.pth",
                    "seed_1_global_step_80000000.pth",
                    "seed_1_global_step_90000000.pth",
                    "seed_1_global_step_100000000.pth",
                    "seed_1_global_step_110000000.pth",
                    "seed_1_global_step_120000000.pth",
                    "seed_1_global_step_130000000.pth",
                    "seed_1_global_step_140000000.pth",
                    "seed_1_global_step_150000000.pth",
                    "seed_1_global_step_160000000.pth",
                    "seed_1_global_step_170000000.pth",
                    "seed_1_global_step_180000000.pth",
                    "seed_1_global_step_190000000.pth",
                    "seed_1_global_step_199990000.pth",
                    ]

    df = pd.DataFrame()
    for model in saved_models:
        print("Model", model)
        episodic_returns = evaluate(
            model_path=f"saved_networks/{model}",
            eval_episode=20,
            Model=QNetwork,
            device=torch.device(device),
        )
        df[model] = episodic_returns
    df.to_csv("episodic_returns.csv", index=False)
