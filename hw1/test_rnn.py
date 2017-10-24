import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils import *
from model import RNNModel
import sys
import pickle

if __name__ == '__main__': 
    norm = pickle.load(open('norm.pkl', 'rb'))
    x_test, test_id = load_test_data('mfcc')
    x_test = x_test / norm
    output = [] 
    
    test_loader = DataLoader(
        dataset=TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(x_test)),
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    
    model = RNNModel(41, 40).cuda()
    model.load_state_dict(torch.load('model/model_rnn.pt'))
    model.eval()

    for i, (x, _) in enumerate(test_loader):
        y = model(Variable(x).cuda())
        y = torch.max(y, -1)[1].data.cpu().numpy()
        y = encode(y)
        output += y

    output_file = open('predict.csv', 'w')
    print('id,phone_sequence', file=output_file)
    for i in range(len(output)):
        print('{},{}'.format(test_id[i], output[i]), file=output_file)
        
        
