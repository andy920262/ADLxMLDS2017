import sys
import pickle
import numpy as np
from utils import *
from model import *
import torch
from torch.autograd import Variable

s = ['klteYv1Uv9A_27_33.avi',
    '5YJaS2Eswg0_22_26.avi',
    'UbmZAe5u5FI_132_141.avi',
    'JntMAcTlOF0_50_70.avi',
    'tJHUH9tpqPg_113_118.avi']

#(x_train, y_train), (x_test, y_test, test_id) =  pickle.load(open('data.pkl', 'rb'))
(x_train, y_train), (x_test, y_test, test_id) = load_data(sys.argv[1])
#pickle.dump(((x_train, y_train), (x_test, y_test, test_id)), open('data.pkl', 'wb')) 
index_word = pickle.load(open('itow.pkl', 'rb'))
input_size = x_train.shape[-1]
output_size = len(index_word)
model = S2VT(input_size, output_size).cuda()
model.load_state_dict(torch.load('model.pt'))
model.eval()
output = model(Variable(torch.FloatTensor(x_test)).cuda())
output = torch.max(output, -1)[1].view(100, -1).cpu().data.numpy()
predict = []
for y in output:
    seq = ' '.join([index_word[i] for i in y]).strip() + '.'
    seq = seq[0].upper() + seq[1:]
    predict.append(seq)
with open(sys.argv[2], 'w+') as out_file:
    for id, pred in zip(test_id, predict):
        if id in s:
            print('{},{}'.format(id, pred), file=out_file)
