import sys
import time
import argparse

import numpy as np
import torch
import editdistance
from torch.autograd import Variable
from progressbar import ProgressBar, ETA, FormatLabel, Bar

from utils import load_train_data, split_validation
from model import RNNModel

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='mfcc')
parser.add_argument('-v', '--valid_rate', default=0.1)
parser.add_argument('-e', '--epochs', default=20)
parser.add_argument('-b', '--batch_size', default=64)
args = parser.parse_args()

dataset = args.dataset
valid_rate = args.valid_rate
epochs = args.epochs
batch_size = args.batch_size

if __name__ == '__main__':
    
    print('Loading train data.')
    x_train, y_train = load_train_data(dataset)
    (x_train, y_train), (x_valid, y_valid) = split_validation(x_train, y_train, valid_rate)
    input_dim = x_train.shape[2]
    output_dim = y_train.shape[2]

    # Convert data from numpy arrray to tensor
    x_train, y_train = torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    x_valid, y_valid = Variable(torch.FloatTensor(x_valid)), Variable(torch.FloatTensor(y_valid))

    print('Initial RNN model.')
    model = RNNModel(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters())
    loss = torch.nn.BCELoss()

    print('Start training.')
    for epoch in range(1, epochs + 1):
        print('Epoch: {}/{}'.format(epoch, epochs))

        widgets = [FormatLabel(''), ' ', Bar('=', '[', ']'), ' - ', ETA(), ' ', FormatLabel('')]
        pbar = ProgressBar(widgets=widgets, maxval=x_train.size(0))
        pbar.start()

        for batch_i in range(0, x_train.size(0), batch_size):
            # Get batch data
            x_batch = Variable(x_train[batch_i : min(batch_i + batch_size, x_train.size(0) - 1)])
            y_batch = Variable(y_train[batch_i : min(batch_i + batch_size, y_train.size(0) - 1)])
            # Optimize
            output = model(x_batch)
            train_loss = loss(output.view(output.size(0) * output.size(1), output.size(2)),
                            y_batch.view(y_batch.size(0) * y_batch.size(1), y_batch.size(2)))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Validation test
            val_out = model(x_valid)
            val_loss = loss(val_out.view(val_out.size(0) * val_out.size(1), val_out.size(2)),
                            y_valid.view(y_valid.size(0) * y_valid.size(1), y_valid.size(2)))
            
            # Update progressbar
            widgets[0] = FormatLabel('{}/{}'.format(batch_i + x_batch.size(0), x_train.size(0)))
            widgets[-1] = FormatLabel(
                    #'train_loss: {}, train_ed: {}, val_loss: {}, val_ed: {}'.format(
                    #    train_loss, train_ed, val_loss, val_ed))
                    'train_loss: {:.4f}, val_loss: {:.4f}'.format(train_loss.data[0], val_loss.data[0]))
            pbar.update(batch_i)

        pbar.finish()

