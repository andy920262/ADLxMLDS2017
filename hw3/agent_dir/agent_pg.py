from agent_dir.agent import Agent
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import random

class Policy(nn.Module):
    '''
    def __init__(self):
        super(Policy, self).__init__()
        self.dense = nn.Linear(6400, 256)
        self.action = nn.Linear(256, 3)
        self.value = nn.Linear(256, 1)
    
    def init_weight(self):
        self.dense.weight.data.normal_(0, (1 / 6400)**0.5)
        self.action.weight.data.normal_(0, (1 / 256)**0.5)
        self.value.weight.data.normal_(0, (1 / 256)**0.5)
        
    def forward(self, input):
        output = F.relu(self.dense(input))
        action = F.softmax(self.action(output))
        value = self.value(output)
        return action, value
    '''
    def __init__(self):
        super(Policy, self).__init__()
        # (1, 80, 80)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2)
        # (8, 39, 39)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        # (32, 19, 19)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        # (32, 9, 9)
        self.dense = nn.Linear(9 * 9 * 16, 256)
        self.action = nn.Linear(256, 3)
        self.value = nn.Linear(256, 1)
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            weight_shape = list(m.weight.data.size())
            fan_in = np.prod(weight_shape[1:4])
            fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            weight_shape = list(m.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)    

    def forward(self, input):
        output = F.relu(self.conv1(input))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = F.relu(self.dense(output.view(output.size(0), -1)))
        action = F.softmax(self.action(output))
        value = self.value(output)
        return action, value
    

class Agent_PG(Agent):
    def __init__(self, env, args, batch_size=256, episode=10000, gamma=0.99, lr=1e-4, eps_max=1.0, eps_decay=15.0, eps_min=0.05):
        super(Agent_PG,self).__init__(env)
        self.batch_size = batch_size
        self.eps_max = eps_max
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.episode = episode
        self.gamma = gamma
        self.memory = []
        self.policy = Policy().cuda()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        #self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr, alpha=0.9)
        self.prev_ob = None
        if args.test_pg:
            print('loading trained model')
            self.policy.load_state_dict(torch.load('model_pg_best.pt'))
            self.policy.eval()
    
    def init_game_setting(self):
        self.prev_ob = None

    def update(self):
        state, reward, action = zip(*self.memory)
        state = torch.stack(state)#.squeeze()
        reward = torch.stack(reward)
        action = torch.stack(action)
        for i in range(reward.size()[0] - 1)[::-1]:
            reward[i, 0] = reward[i, 0] if reward[i, 0] != 0 else reward[i + 1, 0] * self.gamma
        reward = (reward - reward.mean()) / (reward.std() + np.finfo(np.float32).eps)
        for batch_i in range(0, reward.size()[0], self.batch_size):
            batch_s = Variable(state[batch_i : min(batch_i + self.batch_size, reward.size()[0])]).cuda()
            batch_r = Variable(reward[batch_i : min(batch_i + self.batch_size, reward.size()[0])]).cuda()
            batch_a = Variable(action[batch_i : min(batch_i + self.batch_size, reward.size()[0])]).cuda()
            score, value = self.policy(batch_s) 
            score = torch.log(score.gather(1, batch_a))
            self.optimizer.zero_grad()
            value_loss = F.mse_loss(value, batch_r)
            score.backward(-1 * (batch_r - value))
            value_loss.backward()
            self.optimizer.step()
        
        del self.memory[:]

    def preprocess_state(self, state):
        state = state[35:195]
        state = 0.2126 * state[:, :, 0] + 0.7152 * state[:, :, 1] + 0.0722 * state[:, :, 2]
        state = state.astype(np.uint8)[::2, ::2]
        return state.astype(np.float64)

    def train(self):
        best_reward = -7122
        reward_sum = 0
        logfile = open('pg.log', 'w+')
        for e in range(1, self.episode):
            observation = self.env.reset()
            done = False
            self.prev_ob = None
            score = 0
            while not done:
                action, state = self.make_action(observation, False)
                observation, reward, done, _ = self.env.step(action + 1)
                self.memory.append((
                    torch.FloatTensor([state]),
                    torch.FloatTensor([reward]),
                    torch.LongTensor([action])))
                if reward != 0:
                    score += reward
            print('Episode {}, reward {}'.format(e, score))
            print('Episode {}, reward {}'.format(e, score), file=logfile)
            logfile.flush()
            reward_sum += score
            self.update()
            if e % 30 == 0:
                if reward_sum > best_reward:
                    print('Save model. Avg reward = {}'.format(reward_sum / 30))
                    torch.save(self.policy.state_dict(), 'model_pg.pt')
                    best_reward = reward_sum
                reward_sum = 0


    def make_action(self, observation, test=True):
        observation = self.preprocess_state(observation)
        
        if self.prev_ob is None:
            state = np.zeros(observation.shape)
        else:
            state = observation - self.prev_ob
        self.prev_ob = observation
        
        action, value = self.policy(Variable(torch.FloatTensor(state).cuda().view(1, 1, 80, 80)))
        if test:
            return action.max(-1)[1].data[0] + 1
        action = action.multinomial()
        return action.data[0, 0],  state

