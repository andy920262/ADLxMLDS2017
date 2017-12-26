import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, tag_dim, noise_dim):
        super(Generator, self).__init__()
        self.dense1 = nn.Linear(tag_dim + noise_dim, 128 * 4 * 4)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.apply(weights_init)

    def forward(self, tag, noise):
        batch_size = tag.size()[0]
        dense1 = F.relu(self.dense1(torch.cat((tag, noise), -1))).view(batch_size, 128, 4, 4)
        conv1 = F.relu(self.conv1(self.bn1(dense1)))
        conv2 = F.relu(self.conv2(self.bn2(conv1)))
        conv3 = F.relu(self.conv3(self.bn3(conv2)))
        conv4 = self.conv4(self.bn4(conv3))
        return F.sigmoid(conv4.transpose(1, 3))
        
class Discriminator_nobn(nn.Module):
    def __init__(self, tag_dim):
        super(Discriminator_nobn, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.dense1 = nn.Linear(128 * 4 * 4 + tag_dim, 512)
        self.dense2 = nn.Linear(512, 1)
        self.apply(weights_init)

    def forward(self, image, tag):
        batch_size = image.size()[0]
        conv1 = F.relu(self.conv1(image.transpose(1, 3)))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3))
        dense1 = F.relu(self.dense1(torch.cat((conv4.view(batch_size, -1), tag), -1)))
        dense2 = self.dense2(dense1)
        return dense2

class Discriminator(nn.Module):
    def __init__(self, tag_dim):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.dense1 = nn.Linear(128 * 4 * 4 + tag_dim, 512)
        self.dense2 = nn.Linear(512, 1)
        self.apply(weights_init)

    def forward(self, image, tag):
        batch_size = image.size()[0]
        conv1 = F.relu(self.bn1(self.conv1(image.transpose(1, 3))))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        conv4 = F.relu(self.bn4(self.conv4(conv3)))
        dense1 = F.relu(self.dense1(torch.cat((conv4.view(batch_size, -1), tag), -1)))
        dense2 = self.dense2(dense1)
        #return dense2
        return F.sigmoid(dense2)
