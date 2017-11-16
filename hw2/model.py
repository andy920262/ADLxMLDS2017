import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0.2)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        return output, hidden

class Decoder(torch.nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
                input_size=hidden_size * 2,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0.2)
        self.attn = nn.Linear(hidden_size, 80)
        self.attn_c = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.emb = nn.Embedding(output_size, hidden_size)
        self.drop = nn.Dropout(0.2)
        

    def forward(self, input, last_out, hidden, enc_hids=None, attn=False):
        if attn:
            attn_w = F.softmax(self.attn(hidden[0][0])) # (b, 1, 80)
            #score = torch.exp(torch.bmm(enc_hids, hidden[0][0].unsqueeze(-1)).squeeze()) # (b, 80, hid) * (b, hid, 1) = (b, 80, 1)
            #attn_w = score / score.sum()
            attn = torch.bmm(attn_w.unsqueeze(1), enc_hids).squeeze()#(b, 256)
            h = F.tanh(self.attn_c(torch.cat((attn, hidden[0].squeeze()), -1))).unsqueeze(0)#(b, 1, hid)
            output, hidden = self.lstm(torch.cat((input, self.drop(self.emb(last_out))), -1), (h, hidden[1]))
        else:
            output, hidden = self.lstm(torch.cat((input, self.drop(self.emb(last_out))), -1), hidden)
        output = self.out(output)
        return output, hidden
