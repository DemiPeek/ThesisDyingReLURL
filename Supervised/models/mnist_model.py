import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.Conv2d(64, 64, 2),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(23104, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.network(x)
        return F.log_softmax(x)