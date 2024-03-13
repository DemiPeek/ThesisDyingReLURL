import copy
import numpy as np
from replaybuffer import ReplayBuffer
import matplotlib.pyplot as plt
import torch
import random as r
import os
from os import path
from environments.maze_env import Maze
import random


def count_dying_relu(relu_outputs, margin=0.001):
    """
    Counts the "weighted" number of dying ReLUs and completely dead ReLUs for a batch of outputs from a ReLU layer.

    Args:
    - relu_outputs (torch.Tensor): A 2D tensor where the first dimension is the batch size
      and the second dimension is the number of neurons.
    - margin (float): The threshold below which a neuron's output is considered as zero.

    Returns:
    - float: Weighted count of dying ReLUs.
    - int: Number of completely dead ReLUs.
    """

    # Compute a boolean tensor indicating for each neuron and data point if it's dead
    is_dead_per_data_point = (relu_outputs.abs() < margin)

    # Compute the fraction of dead data points for each neuron
    dead_fractions = is_dead_per_data_point.float().mean(dim=0)

    # Weighted count of dying ReLUs
    weighted_count = dead_fractions.sum()
    # Count completely dead ReLUs
    completely_dead = torch.sum(dead_fractions == 1.0).item()

    return weighted_count.item(), completely_dead


def count_time_dead(dead_timecount, reincarnate_type, relu_outputs, margin=0.001):
    # Compute a boolean tensor indicating for each neuron and data point if it's dead
    is_dead_per_data_point = (relu_outputs.abs() < margin)

    # Compute the fraction of dead data points for each neuron
    dead_fractions = is_dead_per_data_point.float().mean(dim=0)

    threshold1 = 49
    threshold2 = 99  # count through 50 iterations after leaky relu threshold

    if reincarnate_type == 'lrelu_inf':
        # apply other activation function for the rest of the iterations
        condition1 = (dead_fractions == 1.0)
        condition2 = (dead_fractions != 1.0) & (dead_timecount >= threshold1)
        condition3 = (dead_fractions != 1.0) & (dead_timecount < threshold1)
        dead_timecount = torch.where(condition1, dead_timecount + 1, dead_timecount)
        dead_timecount = torch.where(condition2, dead_timecount + 1, dead_timecount)
        dead_timecount = torch.where(condition3, 0, dead_timecount)
    elif reincarnate_type == 'lrelu_x':
        condition1 = (dead_fractions == 1.0)
        # dead_fractions != 1.0 to prevent 2 conditions being satisfied sequentially in the same iteration
        condition2 = (dead_fractions != 1.0) & (dead_timecount >= threshold1) & (dead_timecount < threshold2)
        condition3 = (dead_fractions != 1.0) & (dead_timecount >= threshold2)
        condition4 = (dead_fractions != 1.0) & (dead_timecount < threshold1)
        dead_timecount = torch.where(condition1, dead_timecount + 1, dead_timecount)
        dead_timecount = torch.where(condition2, dead_timecount + 1, dead_timecount)
        dead_timecount = torch.where(condition3, 0, dead_timecount)
        dead_timecount = torch.where(condition4, 0, dead_timecount)
    elif reincarnate_type == 'standard':
        dead_timecount = dead_timecount
    else:
        # apply other activation function one iteration
        dead_timecount = torch.where(dead_fractions == 1.0, dead_timecount + 1, 0)

    lrelu_count = torch.where(dead_timecount >= 49, 1, 0)

    return dead_timecount, lrelu_count


def strtobool(v):
  return str(v).lower() in ("yes", "true", "t", "1")


def set_seed(seed: int = 42) -> None:  # int=42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True  # Only this one is necessary for reproducibility
    # torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def to_numpy(tensor):
    if tensor is None:
        return None
    elif tensor.nelement() == 0:
        return np.array([])
    else:
        return tensor.cpu().detach().numpy()


def fill_buffer(buffer, num_transitions, env, noreset=False):
    if noreset:
        dont_take_reward = True
    else:
        dont_take_reward = False
        if env.name == 'maze':
            env.create_map()
            mode=1
        elif env.name == 'catcher':
            mode=-1
        else:
            pass

    end = num_transitions
    i = 0
    while i <= end:
        done = False
        state = env.observe()
        action = env.actions[r.randrange(env.num_actions)]
        reward = env.step(action, dont_take_reward=dont_take_reward)
        next_state = env.observe()
        if env.inTerminalState():
            env.reset(mode=1)
            done = True
        buffer.add(state, action, reward, next_state, done)
        i += 1
        if i >= end and not done:
            i -= 1


def visualize_buffer_batch(agent, steps=10):
    """Visualize what the states in the buffer look like"""
    for i in range(steps):
        STATE, _, _, _, _ = agent.buffer.sample(1)
        STATE = to_numpy(STATE)
        img = plt.imshow(STATE[0][0], cmap='gray')
        # save image as PDF
        # plt.savefig('state_%s.pdf' % i)
        plt.pause(0.5)
        plt.draw()
    plt.close()


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)
