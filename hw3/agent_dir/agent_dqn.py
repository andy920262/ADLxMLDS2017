from agent_dir.agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
import random
import math
import numpy as np

class DQN(nn.Module):
    def __init__(self, duel=False):
        super(DQN, self).__init__()
        self.duel = duel
        # (4, 84, 84)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        # (32, 20, 20)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # (64, 9, 9)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # (64, 7, 7)
        self.dense = nn.Linear(7 * 7 * 64, 512)
        if self.duel:
            self.value = nn.Linear(512, 1) 
            self.advantage = nn.Linear(512, 3) 
        else:
            self.output = nn.Linear(512, 3)
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            weight_shape = list(m.weight.data.size())
            fan_in = np.prod(weight_shape[1:4])
            fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
            #w_bound = np.sqrt(6. / (fan_in + fan_out))
            w_bound = np.sqrt(1. / (fan_in))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            weight_shape = list(m.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            #w_bound = np.sqrt(6. / (fan_in + fan_out))
            w_bound = np.sqrt(1. / (fan_in))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)

    def forward(self, input):
        output = F.relu(self.conv1(input.transpose(3, 2).transpose(2, 1)))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = F.relu(self.dense(output.view(output.size(0), -1)))
        if self.duel:
            value = self.value(output)
            advantage = self.advantage(output)
            return value.expand_as(advantage) + advantage - advantage.mean(-1).unsqueeze(-1).expand_as(advantage)
        else:
            return self.output(output)

class Agent_DQN(Agent):
    def __init__(self, env, args, episode=1000000, lr=1e-4, max_memory=10000, batch_size=32, gamma=0.99, eps_min=0.05, eps_max=1.0, eps_step=1000000):
        super(Agent_DQN,self).__init__(env)
        self.batch_size = batch_size
        self.gamma = gamma
        self.episode = episode
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_step = eps_step
        self.memory = deque(maxlen=max_memory)
        self.targetQ = DQN().cuda()
        self.Q = DQN().cuda()
        self.targetQ.load_state_dict(self.Q.state_dict())
        self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=lr)
        if args.test_dqn:
            self.Q.load_state_dict(torch.load('model_dqn_best.pt'))
            print('loading trained model')

    def init_game_setting(self):
        pass
    
    def get_eps(self, t):
        if t > self.eps_step:
            return 0
        return self.eps_min + (self.eps_max - self.eps_min) * ((self.eps_step - t) / self.eps_step)

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0
        batch = random.sample(self.memory, self.batch_size)
        batch_state, batch_next, batch_action, batch_reward, batch_done = zip(*batch)
        
        batch_state = Variable(torch.stack(batch_state)).cuda().squeeze()
        batch_next = Variable(torch.stack(batch_next)).cuda().squeeze()
        batch_action = Variable(torch.stack(batch_action)).cuda()
        batch_reward = Variable(torch.stack(batch_reward)).cuda()
        batch_done = Variable(torch.stack(batch_done)).cuda()
        
        current_q = self.Q(batch_state).gather(1, batch_action)
        next_q = batch_reward + (1 - batch_done) * self.gamma * self.targetQ(batch_next).detach().max(-1)[0].unsqueeze(-1)
 
        self.optimizer.zero_grad()
        loss = F.mse_loss(current_q, next_q)
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    def train(self):
        logfile = open('simple_dqn.log', 'w+')
        step = 0
        for e in range(1, self.episode):
            state = self.env.reset()
            done = False
            reward_sum = 0
            loss = []
            for i in range(random.randint(1, 30)):
                state, _, _, _ = self.env.step(1)
            state = state.astype(np.float64)
            while not done:
                eps = self.get_eps(step)
                action = random.randint(0, 2) if random.random() < eps else self.make_action(state, False)
                next_state, reward, done, _ = self.env.step(action + 1)
                next_state = next_state.astype(np.float64)
                reward_sum += reward
                step += 1
                self.memory.append((
                    torch.FloatTensor([state]),
                    torch.FloatTensor([next_state]),
                    torch.LongTensor([action]),
                    torch.FloatTensor([reward]),
                    torch.FloatTensor([done])))
                state = next_state
                if step >= 10000 and step % 4 == 0:
                    loss.append(self.update())
                #if step % 1000 == 0:
                    self.targetQ.load_state_dict(self.Q.state_dict())
            print('Epidoe: {}, step={}, eps={:.4f}, loss={:.4f}, reward={}'.format(e, step, eps, np.mean(loss), reward_sum))
            print('Epidoe: {}, step={}, eps={:.4f}, loss={:.4f}, reward={}'.format(e, step, eps, np.mean(loss), reward_sum), file=logfile)
            logfile.flush()
            #if e % 50 == 0:
            #    print('Save model.')
            #    torch.save(self.Q.state_dict(), 'model_dqn.pt')


    def make_action(self, observation, test=True):
        if test:
            return self.Q(Variable(torch.FloatTensor(observation).unsqueeze(0)).cuda()).max(-1)[1].data[0] + 1
        return self.Q(Variable(torch.FloatTensor(observation).unsqueeze(0)).cuda()).max(-1)[1].data[0]
