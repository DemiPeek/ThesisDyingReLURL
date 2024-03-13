import os
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tyro


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

    def relu(self, x):
        return self.network[:-1](x / 255.0)


def evaluate():
    # env setup
    envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i) for i in range(args.num_envs)]
        )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    model = QNetwork(envs).to(device)
    model.load_state_dict(torch.load(f'saved_networks/seed_1_global_step_199990000.pth', map_location=device))
    model.eval()

    obs, _ = envs.reset()
    relu_output = model.relu((torch.Tensor(obs).to(device))).detach().numpy()
    relu_output = relu_output.reshape((16, 32))

    df = pd.DataFrame(relu_output)
    df_rounded = df.round(2)

    colors_array = np.where(df_rounded != 0, 'lightgreen', 'white')
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    table = ax.table(cellText=df_rounded.values,
                     cellColours=colors_array,
                     loc='center')

    plt.savefig('table_image_colored2.png', bbox_inches='tight', pad_inches=0.5)
    plt.show()

    return


if __name__ == "__main__":
    from dqn_atari import make_env
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    evaluate()
