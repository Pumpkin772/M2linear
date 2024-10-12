
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from tensorboardX import SummaryWriter
from collections import deque
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

class Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Model, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.feature = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU() )
        self.advantage = nn.Sequential(nn.Linear(128, 128),nn.ReLU(),nn.Linear(128, action_dim))

        self.value = nn.Sequential(nn.Linear(128, 128),nn.ReLU(),nn.Linear(128, 1))

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def greedy_act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0].item()
        # action = action.detach().cpu().numpy()
        else:
            action = random.randrange(self.action_dim)
        return action

    def online_act(self, state):
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        q_value = self.forward(state)
        action = q_value.max(1)[1].data[0].item()
        return action

class Dueling_DQN():
    def __init__(self, state_dim, action_dim):
        self.Qnet= Model(state_dim,action_dim).to(device)
        self.target_Qnet = Model(state_dim,action_dim).to(device)
        self.target_Qnet.load_state_dict(self.Qnet.state_dict())
        self.lr = 3e-4
        self.optimizer = optim.Adam(self.Qnet.parameters(), lr=self.lr)
        self.replay_buffer_size = 10000
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.gamma = 0.99


    def train(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        q_values = self.Qnet(state)
        next_q_values = self.target_Qnet(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def update_target(self):
        self.target_Qnet.load_state_dict(self.Qnet.state_dict())

    def save_model(self,path):
        torch.save(self.Qnet.state_dict(), path + 'qnet.pth')

    def load_model(self,path):
        self.Qnet.load_state_dict(torch.load(path + 'qnet.pth'))
        self.target_Qnet.load_state_dict(torch.load(path + 'qnet.pth'))


class MADDQN():
    def  __init__(self, state_dim, action_dim, n_agents):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.model = [Dueling_DQN(self.state_dim, self.action_dim) for i1 in range(self.n_agents)]

    def train(self, batch_size):
        loss_list = []
        for agent_id in range(self.n_agents):
            loss = self.model[agent_id].train(batch_size)
            loss_list.append(loss)
        return np.sum(loss_list)

    def update_target(self):
        for agent_id in range(self.n_agents):
            self.model[agent_id].target_Qnet.load_state_dict(self.model[agent_id].Qnet.state_dict())

    def save_model(self, path):
        for agent_id in range(self.n_agents):
            torch.save(self.model[agent_id].Qnet.state_dict(), path + 'agent'+str(agent_id)+'_qnet.pth')
        print('=============The model is saved=============')

    def load_model(self, path):
        for agent_id in range(self.n_agents):
            self.model[agent_id].Qnet.load_state_dict(torch.load(path + 'agent'+str(agent_id)+'_qnet.pth'))
            self.model[agent_id].target_Qnet.load_state_dict(torch.load(path + 'agent'+str(agent_id)+'_qnet.pth'))