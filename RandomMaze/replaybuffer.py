import numpy as np
import torch

# Buffer structure adopted from Denis Yarats, SAC-AE (https://github.com/denisyarats/pytorch_sac_ae)


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.obses = np.empty((self.capacity, self.obs_shape[0], self.obs_shape[1], self.obs_shape[2]), dtype=np.float32)
        self.next_obses = np.empty((self.capacity, self.obs_shape[0], self.obs_shape[1], self.obs_shape[2]), dtype=np.float32)
        self.actions = np.empty((self.capacity, self.action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=bool)

        self.idx = 0
        self.full = False

        self.id = None
        self.count = 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    # Write a function that empties the replaybuffer, but leaves 32 samples in the buffer
    def empty(self):
        self.obses = np.empty((self.capacity, self.obs_shape[0], self.obs_shape[1], self.obs_shape[2]), dtype=np.float32)
        self.next_obses = np.empty((self.capacity, self.obs_shape[0], self.obs_shape[1], self.obs_shape[2]), dtype=np.float32)
        self.actions = np.empty((self.capacity, self.action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=bool)
        self.idx = 0
        self.full = False
        self.id = None
        self.count = 0

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        dones = torch.as_tensor(self.dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, dones

    def sample_trajectory(self):
        # TODO check what happens if we only sample on-policy trajectories for the entropy loss
        self.count += 1
        if self.id is None:
            self.id = np.where(self.dones[0:self.idx])[0]  # Find the end of all the trajectories (dones=True)
        if self.count % 1000 == 0:
            self.id = np.where(self.dones[0:self.idx])[0]

        random_start = np.random.randint(low=0, high=len(self.id)-2, size=1)              # Find the start of a trajectory in the buffer
        random_start_id = self.id[random_start]+1
        random_end_id = self.id[random_start + 1] # new trajectory starts 1 past the done index
        if random_end_id - random_start_id < 3:
            while random_end_id - random_start_id < 3:
                random_start = np.random.randint(low=0, high=len(self.id)-2, size=1)              # Find the start of a trajectory in the buffer
                random_start_id = self.id[random_start]+1
                random_end_id = self.id[random_start + 1]
        trajectory_ids = np.arange(random_start_id, random_end_id)  # Make all the ids

        obses = torch.as_tensor(self.obses[trajectory_ids], device=self.device).float()

        return obses
