import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(RNNModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.dnn = nn.Linear(hidden_size, output_size)
    
    def forward(self, input):
        hidden = (Variable(torch.zeros(self.input_size, self.hidden_size)),
                  Variable(torch.zeros(self.input_size, self.hidden_size)))
        output = []

        for i, x in enumerate(input.chunk(input.size(1), dim=1)):
            # x: (batch_size, 1, input_size)
            output_i, hidden = self.rnn(x, hidden)

            # output_i: (batch_size, 1, hidden_size)
            output_i = output_i.view(output_i.size(0), output_i.size(2))
            
            # output_i: (batch_size, hidden_size)
            output_i = F.sigmoid(self.dnn(output_i))
            output += [output_i]
        
        return torch.stack(output, dim=1).squeeze(dim=2)

