import torch
import torch.nn as nn
from torch.autograd import Variable

    
class S2VT(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(S2VT, self).__init__()
        self.lstm1 = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True)
        self.lstm2 = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.output_size = output_size

    def forward(self, input):
        batch_size = input.size(0)
        dec_pad = Variable(torch.zeros(batch_size, 1, 4096)).cuda()

        # Encode
        lstm1_out, lstm1_hid = self.lstm1(input, None)
        _, lstm2_hid = self.lstm2(lstm1_out, None)

        # Decode
        output = []
        lstm1_out, lstm1_hid = self.lstm1(dec_pad, lstm1_hid)
        lstm2_out = Variable(torch.zeros(batch_size, 1, 256)).cuda()
        for i in range(40):
            lstm1_out, lstm1_hid = self.lstm1(dec_pad, lstm1_hid)
            lstm2_out, lstm2_hid = self.lstm2(lstm1_out + lstm2_out, lstm2_hid)
            output += [lstm2_out]
        
        output = torch.stack(output, 1).contiguous().view(-1, 256)
        output = self.output_layer(output)#.view(batch_size, 40, -1)
        return output
