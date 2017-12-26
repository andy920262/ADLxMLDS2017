import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from progressbar import ProgressBar, ETA, FormatLabel, Bar
from utils import *
from model import *
import random
import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--noise_dim', type=int, default=100)
parser.add_argument('--g_update', type=int, default=1)
parser.add_argument('--preload', type=bool, default=True)
parser.add_argument('--model_dir', default='./model')
parser.add_argument('--log_file', default='model.log')
arg = parser.parse_args()

tag_dim = 25

def get_wrong_tag(tag):
    wrong = torch.zeros(tag.size())
    for i, t in enumerate(tag):
        j = random.randint(0, 12)
        while t[j] == 1:
            j = random.randint(0, 12)
        wrong[i][j] = 1
        j = random.randint(13, 24)
        while t[j] == 1:
            j = random.randint(13, 24)
        wrong[i][j] = 1
    return wrong

if __name__ == '__main__':
    print('Load data.')
    if arg.preload:
        x_train, y_train = pickle.load(open('data.pkl', 'rb'))
    else:
        x_train, y_train = load_train_data('data')
        pickle.dump((x_train, y_train), open('data.pkl', 'wb'))
    y_train = y_train.astype(np.float64)

    train_loader = DataLoader(
            dataset=TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train)),
            batch_size=arg.batch_size,
            shuffle=True,
            num_workers=4)
    
    print('Initial model.')
    g_net = Generator(tag_dim, arg.noise_dim).cuda()
    d_net = Discriminator(tag_dim).cuda()
    g_optimizer = torch.optim.Adam(g_net.parameters(), lr=arg.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(d_net.parameters(), lr=arg.lr, betas=(0.5, 0.999))
    
    log_file = open(arg.log_file, 'w')

    for epoch in range(1, arg.epochs + 1):
        
        print('Epoch: {}/{}'.format(epoch, arg.epochs))
        g_total_loss, d_total_loss = 0, 0
        
        widgets = [FormatLabel(''), ' ', Bar('=', '[', ']'), ' - ', ETA(), ' ', FormatLabel('')]
        pbar = ProgressBar(widgets=widgets, maxval=x_train.shape[0])
        pbar.start()


        for i, (real_img, real_tag) in enumerate(train_loader):
            for p in d_net.parameters():
                p.requires_grad = True
            
            noise = Variable(torch.randn(real_img.size()[0], arg.noise_dim), volatile = True).cuda()
            wrong_tag = Variable(get_wrong_tag(real_tag)).cuda()
            real_img = Variable(real_img).cuda()
            real_tag = Variable(real_tag).cuda()
            fake_img = Variable(g_net(real_tag, noise).data)

            # Update D
            d1 = d_net(real_img, real_tag)
            d2 = d_net(fake_img, real_tag)
            d3 = d_net(real_img, wrong_tag)
            
            d_loss = F.binary_cross_entropy(d1, Variable(torch.ones(d1.size())).cuda()) + \
                    F.binary_cross_entropy(d2, Variable(torch.zeros(d2.size())).cuda()) / 2 + \
                    F.binary_cross_entropy(d3, Variable(torch.zeros(d3.size())).cuda()) / 2
            
            if i % arg.g_update == 0:
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
            
            d_total_loss += d_loss.data[0]

            # Update G
            for p in d_net.parameters():
                p.requires_grad = False
            noise = Variable(torch.randn(real_img.size()[0], arg.noise_dim)).cuda()
            fake_img = g_net(real_tag, noise)
            g = d_net(fake_img, real_tag)

            g_loss = F.binary_cross_entropy(g, Variable(torch.ones(g.size())).cuda())

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            g_total_loss += g_loss.data[0]

            # Update Progressbar
            widgets[0] = FormatLabel('{}/{}'.format((i * arg.batch_size + real_img.size(0)), x_train.shape[0]))
            widgets[-1] = FormatLabel('g_loss:{:.4f}, d_loss:{:.4f}'.format(
                            g_total_loss / (i + 1), d_total_loss / (i + 1)))

            pbar.update(i * arg.batch_size + real_img.size(0))

        pbar.finish()
        
        print('epoch:{} g_loss:{:.4f} d_loss:{:.4f}'.format(epoch, g_total_loss / (i + 1), d_total_loss / (i + 1)), file=log_file)
        log_file.flush()

        torch.save(g_net.state_dict(), os.path.join(arg.model_dir, 'generator_e{}.pt'.format(epoch)))
        if epoch % 10 == 0:
            torch.save(g_net.state_dict(), os.path.join(arg.model_dir, 'discriminator_e{}.pt'.format(epoch)))

