import sys
import time
import argparse
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from progressbar import ProgressBar, ETA, FormatLabel, Bar

from utils import *
from model import CNNModel

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='mfcc')
parser.add_argument('-v', '--valid_rate', default=0.1)
parser.add_argument('-e', '--epochs', default=0)
parser.add_argument('-b', '--batch_size', default=64)
args = parser.parse_args()

dataset = 'mfcc'
valid_rate = 0.1
epochs = 500
batch_size = 64

torch.manual_seed(87)

if __name__ == '__main__':
    
    print('Loading train data.')
    x_train, y_train = load_train_data(dataset)
    #x_train, y_train = pickle.load(open('data.pkl', 'rb'))
    norm = pickle.load(open('norm.pkl', 'rb'))
    x_train = x_train / norm

    input_dim = x_train.shape[-1]
    output_dim = 40

    (x_train, y_train), (x_valid, y_valid) = split_validation(x_train, y_train, valid_rate)
    
    # Set dataloader
    train_loader = DataLoader(
        dataset=TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    valid_loader = DataLoader(
        dataset=TensorDataset(torch.FloatTensor(x_valid), torch.LongTensor(y_valid)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
        
    print('Initial CNN model.')
    model = CNNModel(input_dim, output_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()

    print('Start training.')

    best_ed = 999
    early_stop_cnt = 0

    for epoch in range(1, epochs + 1):
        
        print('Epoch: {}/{}'.format(epoch, epochs))

        total_loss, total_acc, nonzeros = 0, 0, 0
        
        widgets = [FormatLabel(''), ' ', Bar('=', '[', ']'), ' - ', ETA(), ' ', FormatLabel('')]
        pbar = ProgressBar(widgets=widgets, maxval=x_train.shape[0])
        pbar.start()
        
        for i, (x_batch, y_batch) in enumerate(train_loader):
            # Tensor to variable
            x_batch = Variable(x_batch).cuda()
            y_batch = Variable(y_batch).cuda()

            # Optimize
            output = model(x_batch)
            loss = loss_func(output.view(-1, output.size(-1)), y_batch.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            y_pred = torch.max(output, -1)[1]
            y_pred = y_pred.masked_fill_((y_batch == 0), 0)
            nonzeros += (y_batch != 0).data.sum()
            
            total_loss += loss.data[0]
            total_acc += (y_pred == y_batch).data.sum() - (y_batch == 0).data.sum()

            
            # Update progressbar
            widgets[0] = FormatLabel('{}/{}'.format((i * batch_size + x_batch.size(0)), x_train.shape[0]))
            widgets[-1] = FormatLabel('train_loss:{:.4f}, train_acc:{:.4f}'.format(
                            total_loss / (i + 1), total_acc / nonzeros))

            pbar.update(i * batch_size + x_batch.size(0))
        pbar.finish()
        
        model.eval()
        # Validation test
        total_loss, total_acc, total_ed, nonzeros = 0, 0, 0, 0
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            x_batch = Variable(x_batch).cuda()
            y_batch = Variable(y_batch).cuda()
            
            output = model(x_batch)
            loss = loss_func(output.view(-1, output.size(-1)), y_batch.view(-1))
            
            y_pred = torch.max(output, -1)[1]
            y_pred = y_pred.masked_fill_((y_batch == 0), 0)
            nonzeros += (y_batch != 0).data.sum()
            
            total_loss += loss.data[0]
            total_acc += (y_pred == y_batch).data.sum() - (y_batch == 0).data.sum()
            
            total_ed += avg_ed(encode(y_pred.data.cpu().numpy()), encode(y_batch.data.cpu().numpy()))

        print('Validation: loss:{:.4f}, acc:{:.4f}, ed:{:.4f}'.format(
            total_loss / (i + 1), total_acc / nonzeros, total_ed / y_valid.shape[0]))
             
        early_stop_cnt += 1
        if (total_ed / y_valid.shape[0]) < best_ed:
            early_stop_cnt = 0
            best_ed = total_ed / y_valid.shape[0]
            print('Save best model: ed={:.4f}'.format(best_ed))
            with open('model/model_best.pt'.format(best_ed), 'wb') as file:
                torch.save(model.state_dict(), file)

        if early_stop_cnt >= 20 and best_ed < 15:
            print('No improvement for 20 epochs. Stop training.')
            break
