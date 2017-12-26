from utils import *
from model import *
import torch
from torch.autograd import Variable
import scipy.misc 
import numpy as np
import sys
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=777)
parser.add_argument('--noise_dim', type=int, default=100)
parser.add_argument('--model', default='generator_best.pt')
parser.add_argument('--tag_list', default='tag_list.txt')
parser.add_argument('--n_img', type=int, default=5)
parser.add_argument('--output_dir', default='samples')
parser.add_argument('--gif', action='store_true')
parser.add_argument('--gif_model_dir')
parser.add_argument('--gif_frames', type=int, default=100)
parser.add_argument('--gif_frameskip', type=int, default=1)
parser.add_argument('--cat_img', action='store_true')

arg = parser.parse_args()

tag_dim = 25
torch.manual_seed(arg.seed)

if __name__ == '__main__':
    g_net = Generator(tag_dim, arg.noise_dim).cuda().eval()
    fd = open(arg.tag_list)
    for line in fd:
        tag_id = line.split(',')[0]
        tag = [[0 if line.split(',')[-1].find(c) == -1 else 1 for c in itoc]] * arg.n_img
        tag = np.array(tag).astype(np.float64)
        tag = Variable(torch.FloatTensor(tag)).cuda()
        noise = Variable(torch.FloatTensor(arg.n_img, arg.noise_dim).uniform_(-1, 1)).cuda()
        if not arg.gif:
            g_net.load_state_dict(torch.load(arg.model))
            if arg.cat_img:
                img = np.hstack(g_net(tag, noise).data.cpu().numpy())
                scipy.misc.imsave(os.path.join(arg.output_dir, 'sample_{}.jpg'.format(tag_id)), img)
            else:
                img = g_net(tag, noise).data.cpu().numpy()
                for i, x in enumerate(img):
                    scipy.misc.imsave(os.path.join(arg.output_dir, 'sample_{}_{}.jpg'.format(tag_id, i + 1)), x)
        else:
            import imageio
            img = []
            for i in range(1, arg.gif_frames + 1, arg.gif_frameskip):
                g_net.load_state_dict(torch.load(os.path.join(arg.gif_model_dir, 'generator_e{}.pt'.format(i))))
                img.append(g_net(tag, noise).data.cpu().numpy())
            for _ in range(10):
                img.append(img[-1])
            if arg.cat_img:
                img = [np.hstack(i) for i in img]
                imageio.mimsave(os.path.join(arg.output_dir, 'sample_{}.gif'.format(tag_id)), img)
            else:
                img = np.array(img).transpose(1, 0, 2, 3, 4)
                for i, x in enumerate(img):
                    imageio.mimsave(os.path.join(arg.output_dir, 'sample_{}_{}.gif'.format(tag_id, i + 1)), x)
            


    
