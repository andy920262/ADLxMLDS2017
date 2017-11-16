import sys
import pickle
import math
import random
import numpy as np
from utils import *
from model import *
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from itertools import groupby 

from bleu_eval import BLEU

data_dir = sys.argv[1]
epochs = 100
batch_size = 64
hidden_size = 512
use_attn = True
max_seq = 10


eps_max = 1.0
eps_decay = 10
eps_min = 0.5

def train(encoder, decoder, enc_optim, dec_optim, loss_func, x_train, y_train, eps):
    
    x_train = Variable(torch.FloatTensor(x_train)).cuda()
    y_train = Variable(torch.LongTensor(y_train)).cuda()

    enc_hid = (Variable(torch.zeros(1, x_train.size()[0], hidden_size)).cuda(),
                Variable(torch.zeros(1, x_train.size()[0], hidden_size)).cuda())
    dec_hid = (Variable(torch.zeros(1, x_train.size()[0], hidden_size)).cuda(),
                Variable(torch.zeros(1, x_train.size()[0], hidden_size)).cuda())
    dec_pad = Variable(torch.LongTensor(np.zeros((x_train.size()[0], 1), dtype=np.int64))).cuda()
    enc_pad = Variable(torch.zeros(x_train.size()[0], 1, 4096)).cuda()
    enc_hids = []
    loss = 0

    for t in range(80):
        enc_out, enc_hid = encoder(x_train[:,t,:].unsqueeze(1), enc_hid)
        dec_out, dec_hid = decoder(enc_out, dec_pad, dec_hid)
        enc_hids += [enc_hid[0][0]]
        #enc_hids += [enc_out.squeeze()]
    enc_hids = torch.stack(enc_hids, 1) # (batch, 80, hidden)
    
    dec_out = Variable(torch.LongTensor(np.zeros((x_train.size()[0], 1), dtype=np.int64))).cuda()
    output = []
    loss_penalty = 1
    for t in range(max_seq):
        enc_out, enc_hid = encoder(enc_pad, enc_hid)
        dec_out, dec_hid = decoder(enc_out, dec_out, dec_hid, enc_hids, use_attn)
        loss += loss_penalty * loss_func(dec_out.view(-1, output_size), y_train[:,t])
        #loss_penalty *= 1.05
        dec_out = y_train[:,t].unsqueeze(-1) if random.random() < eps else torch.max(dec_out, -1)[1]
    '''
        output += [dec_out]
    output = torch.stack(output, 1).contiguous().view(-1, output_size)
    
    output = torch.max(output, -1)[1].view(x_train.size()[0], -1).cpu().data.numpy()
    print(' '.join([index_word[i] for i in output[0]]).strip() + '.')
    print(' '.join([index_word[i] for i in y_batch[0]]).strip() + '.')
    '''
    enc_optim.zero_grad()
    dec_optim.zero_grad()
    loss.backward()
    enc_optim.step()
    dec_optim.step()

    return loss.data[0]

def evaluate(input, encoder, decoder):    
    input = Variable(torch.FloatTensor(input)).cuda()

    enc_hid, dec_hid = None, None
    dec_pad = Variable(torch.LongTensor(np.zeros((input.size()[0], 1), dtype=np.int64))).cuda()
    enc_pad = Variable(torch.zeros(input.size()[0], 1, 4096)).cuda()
    enc_hids = []

    for t in range(80):
        enc_out, enc_hid = encoder(input[:,t,:].unsqueeze(1), enc_hid)
        dec_out, dec_hid = decoder(enc_out, dec_pad, dec_hid)
        enc_hids += [enc_hid[0][0]]
        #enc_hids += [enc_out.squeeze()]
    enc_hids = torch.stack(enc_hids, 1) # (batch, 80, hidden)

    dec_out = Variable(torch.LongTensor(np.zeros((input.size()[0], 1), dtype=np.int64))).cuda()
    output = []
    for t in range(max_seq):
        enc_out, enc_hid = encoder(enc_pad, enc_hid)
        dec_out, dec_hid = decoder(enc_out, dec_out, dec_hid, enc_hids, use_attn)
        dec_out = torch.max(dec_out, -1)[1]
        output += [dec_out]
    output = torch.stack(output, 1).contiguous().view(input.size()[0], -1).cpu().data.numpy()
    

    return output

if __name__ == '__main__':
    print('Loading data.')
    #(x_train, y_train), (x_test, y_test, test_id) =  pickle.load(open('data.pkl', 'rb'))
    (x_train, y_train), (x_test, y_test, test_id) = load_data(data_dir)
    #pickle.dump(((x_train, y_train), (x_test, y_test, test_id)), open('data.pkl', 'wb')) 
    index_word = pickle.load(open('itow.pkl', 'rb'))

    input_size = x_train.shape[-1]
    output_size = len(index_word)
    print('Voc size:', output_size)

    print('Initial model.')

    encoder = Encoder(input_size, hidden_size).cuda()
    decoder = Decoder(hidden_size, output_size).cuda()
    enc_optim = torch.optim.RMSprop(encoder.parameters(), lr=0.001, alpha=0.9)
    dec_optim = torch.optim.RMSprop(decoder.parameters(), lr=0.001, alpha=0.9)
    loss_func = torch.nn.CrossEntropyLoss()
    best_bleu = -1

    print('Start training')
    for epoch in range(1, epochs + 1):
        total_loss = 0
        eps = max(0.74, eps_min + (eps_max - eps_min) * (eps_decay / (eps_decay + math.exp((epoch - 1) / eps_decay))))
        print('eps:', eps)
        for batch_i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[batch_i : min(batch_i + batch_size, x_train.shape[0])]
            y_batch = []
            for y in y_train[batch_i : min(batch_i + batch_size, x_train.shape[0])]:
                y_batch.append(y[int(epoch / 1) % y.shape[0]])
            y_batch = np.array(y_batch).astype(np.int64)
                
            total_loss += train(encoder, decoder, enc_optim, dec_optim, loss_func, x_batch, y_batch, eps)
        print('Epoch: {}, loss: {:.4f}'.format(epoch, total_loss / 1450))
        
        # Test
        output = evaluate(x_test, encoder, decoder)
        predict = []
        for y in output:
            seq = []
            for w in y:
                if len(seq) == 0 or w != seq[-1]:
                    seq.append(w)
            seq = ' '.join([index_word[i] for i in seq]).strip()
            seq = seq[0].upper() + seq[1:]
            predict.append(seq)
        total_bleu = 0
        for p, y in zip(predict, y_test):
            y_bleu = 0
            for c in y:
                y_bleu += BLEU(p, c.rstrip('.'))
            total_bleu += y_bleu / len(y)
        print('AVG BLEU = {}'.format(total_bleu / 100))
        with open('ans.txt', 'w+') as out_file:
            for id, pred in zip(test_id, predict):
                print('{},{}'.format(id, pred), file=out_file)
        if total_bleu > best_bleu:
            print('Save model.')
            best_bleu = total_bleu
            with open('encoder.pt', 'wb') as file:
                torch.save(encoder.state_dict(), file)
            with open('decoder.pt', 'wb') as file:
                torch.save(decoder.state_dict(), file)
        
    

