import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings
import numpy as np
from torch.distributions.categorical import Categorical
warnings.filterwarnings("ignore")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.envs = envs
        self.al1 = layer_init(nn.Linear(512 , 64), std=0.01)
        self.al2 = layer_init(nn.Linear(64, 6), std=0.01)
        self.cl1 = layer_init(nn.Linear(512, 64), std=1)
        self.cl2 = layer_init(nn.Linear(64, 1), std=1)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.al2(F.relu(self.al1(hidden)))
        probs = Categorical(logits=logits)  # sigmoid
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.cl2(F.relu(self.cl1(hidden)))
    
    def get_value(self, x):
        hidden = self.network(x / 255.0)
        return self.cl2(F.relu(self.cl1(hidden)))

