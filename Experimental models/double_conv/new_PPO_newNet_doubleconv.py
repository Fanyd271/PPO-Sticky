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
        self.infer_last = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 1)),
            nn.ReLU(),
        )
        self.envs = envs
        self.actor = layer_init(nn.Linear(512 + 1 , 6), std=0.01)
        self.critic = layer_init(nn.Linear(512 + 1, 1), std=1)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        last_action = self.infer_last(x / 255.0)
        sta_act_pair = torch.cat([hidden, last_action], 1)
        logits = self.actor(sta_act_pair)
        probs = Categorical(logits=logits)  # sigmoid
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(sta_act_pair)
    
    def get_value(self, x):
        hidden = self.network(x / 255.0)
        last_action = self.infer_last(x / 255.0)
        sta_act_pair = torch.cat([hidden, last_action], 1)
        return self.critic(sta_act_pair)

