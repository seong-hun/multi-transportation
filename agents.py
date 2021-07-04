import numpy as np
import random
from collections import deque

import torch
import torch.optim as optim
import torch.nn as nn

import config
from utils import hardupdate, softupdate


class ActorNet(nn.Module):
    def __init__(self, cfg):
        super(ActorNet, self).__init__()
        self.lin1 = nn.Linear(cfg.ddpg.state_dim, 32)
        self.lin2 = nn.Linear(32, 64)
        self.lin3 = nn.Linear(64, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, 32)
        self.lin6 = nn.Linear(32, 16)
        self.lin7 = nn.Linear(16, cfg.ddpg.action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.cfg = cfg

    def forward(self, state):
        x1 = self.relu(self.lin1(state))
        x2 = self.relu(self.lin2(x1))
        x3 = self.relu(self.lin3(x2))
        x4 = self.relu(self.lin4(x3))
        x5 = self.relu(self.lin5(x4))
        x6 = self.relu(self.lin6(x5))
        x7 = self.tanh(self.lin7(x6))
        x_scaled = x7 * self.cfg.ddpg.action_scaling \
            + self.cfg.ddpg.action_bias
        return x_scaled


class CriticNet(nn.Module):
    def __init__(self, cfg):
        super(CriticNet, self).__init__()
        self.lin1 = nn.Linear(cfg.ddpg.state_dim+cfg.ddpg.action_dim, 32)
        self.lin2 = nn.Linear(32, 64)
        self.lin3 = nn.Linear(64, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, 32)
        self.lin6 = nn.Linear(32, 16)
        self.lin7 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, state_w_action):
        x1 = self.relu(self.lin1(state_w_action))
        x2 = self.relu(self.lin2(x1))
        x3 = self.relu(self.lin3(x2))
        x4 = self.relu(self.lin4(x3))
        x5 = self.relu(self.lin5(x4))
        x6 = self.relu(self.lin6(x5))
        x7 = self.relu(self.lin7(x6))
        return x7


class DDPG:
    def __init__(self, cfg):
        self.memory = deque(maxlen=cfg.ddpg.memory_size)
        self.behavior_actor = ActorNet(cfg).float()
        self.behavior_critic = CriticNet(cfg).float()
        self.target_actor = ActorNet(cfg).float()
        self.target_critic = CriticNet(cfg).float()
        self.actor_optim = optim.Adam(
            self.behavior_actor.parameters(), lr=cfg.ddpg.actor_lr
        )
        self.critic_optim = optim.Adam(
            self.behavior_critic.parameters(), lr=cfg.ddpg.critic_lr
        )
        self.mse = nn.MSELoss()
        hardupdate(self.target_actor, self.behavior_actor)
        hardupdate(self.target_critic, self.behavior_critic)
        self.cfg = cfg

    def get_action(self, state, net="behavior"):
        with torch.no_grad():
            action = self.behavior_actor(torch.FloatTensor(state)) \
                if net == "behavior" \
                else self.target_actor(torch.FloatTensor(state))
        return np.array(np.squeeze(action))

    def memorize(self, item):
        self.memory.append(item)

    def get_sample(self):
        sample = random.sample(self.memory, self.cfg.ddpg.batch_size)
        state, action, reward, state_next, epi_done = zip(*sample)
        x = torch.tensor(state, requires_grad=True).float()
        u = torch.tensor(action, requires_grad=True).float()
        r = torch.tensor(reward, requires_grad=True).float()
        xn = torch.tensor(state_next, requires_grad=True).float()
        done = torch.tensor(epi_done, requires_grad=True).float().view(-1, 1)
        return x, u, r, xn, done

    def train(self):
        x, u, r, xn, done = self.get_sample()
        with torch.no_grad():
            action = self.target_actor(xn)
            Qn = self.target_critic(torch.cat([xn, action], 1))
            target = r + (1-done) * self.cfg.ddpg.discount * Qn
        Q_w_noise_action = self.behavior_critic(torch.cat([x, u], 1))
        self.critic_optim.zero_grad()
        critic_loss = self.mse(Q_w_noise_action, target)
        critic_loss.backward()
        self.critic_optim.step()

        action_wo_noise = self.behavior_actor(x)
        Q = self.behavior_critic(torch.cat([x, action_wo_noise], 1))
        self.actor_optim.zero_grad()
        actor_loss = torch.sum(-Q)
        actor_loss.backward()
        self.actor_optim.step()

        softupdate(
            self.target_actor,
            self.behavior_actor,
            self.cfg.ddpg.softupdate)
        softupdate(
            self.target_critic,
            self.behavior_critic,
            self.cfg.ddpg.softupdate)

    def save_parameters(self, path_save):
        torch.save({
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'behavior_actor': self.behavior_actor.state_dict(),
            'behavior_critic': self.behavior_critic.state_dict()
        }, path_save)
