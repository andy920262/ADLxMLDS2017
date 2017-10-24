import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=250):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.rnn1 = nn.LSTM(input_size, hidden_size, 4, batch_first=True, bidirectional=True, dropout=0.2)
        self.dnn1 = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, input):
        data_size = input.size() # (batch_size, time_step, input_size)
        
        # rnn1_in: (batch_size, time_step, hidden_size)
        rnn1_out, hidden = self.rnn1(input, None)

        # dnn1_in: (batch_size * time_step, hidden_size)
        dnn1_in = rnn1_out.contiguous().view(-1, rnn1_out.size(-1))
        dnn1_out = self.dnn1(dnn1_in)
        
        output = dnn1_out
        output = output.contiguous().view(data_size[0], data_size[1], output.size(-1))

        return output

class CNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(CNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.cnn1 = nn.Conv1d(input_size, 64, kernel_size=7, stride=1, dilation=1, padding=3)
        self.cnn2 = nn.Conv1d(64, 64, kernel_size=7, stride=1, dilation=1, padding=3)
        #self.cnn3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, dilation=1, padding=2)
        #self.cnn4 = nn.Conv1d(64, 128, kernel_size=5, stride=1, dilation=1, padding=2)
        self.rnn1 = nn.GRU(64, hidden_size, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.dnn1 = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, input):
        data_size = input.size() # (batch_size, time_step, input_size)
       
        # cnn1_in: (batch_size, input_size, time_step)
        cnn1_in = input.transpose(1, 2)
        cnn1_out = self.cnn1(cnn1_in)
        cnn2_out = self.cnn2(cnn1_out)
        #cnn3_out = self.cnn3(cnn2_out)
        #cnn4_out = self.cnn4(cnn3_out)

        # rnn1_in: (batch_size, time_step, hidden_size)
        rnn1_in = cnn2_out.transpose(1, 2)
        rnn1_out, hidden = self.rnn1(rnn1_in, None)

        # dnn1_in: (batch_size * time_step, hidden_size)
        dnn1_in = rnn1_out.contiguous().view(-1, rnn1_out.size(-1))
        dnn1_out = self.dnn1(dnn1_in)
        
        output = dnn1_out
        output = output.contiguous().view(data_size[0], data_size[1], output.size(-1))

        return output


