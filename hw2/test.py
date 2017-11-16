import sys
import pickle
import numpy as np
from utils import *
from model import *
import torch
from torch.autograd import Variable

hidden_size=512
max_seq=10
use_attn=True
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

(x_test, test_id), (x_peer, peer_id) = load_test_data(sys.argv[1])

index_word = pickle.load(open('itow.pkl', 'rb'))
input_size = x_test.shape[-1]
output_size = len(index_word)
encoder = Encoder(input_size, hidden_size).cuda()
decoder = Decoder(hidden_size, output_size).cuda()
encoder.eval()
decoder.eval()
encoder.load_state_dict(torch.load('model/encoder.pt'))
decoder.load_state_dict(torch.load('model/decoder.pt'))

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
with open(sys.argv[2], 'w+') as out_file:
    for id, pred in zip(test_id, predict):
        print('{},{}'.format(id, pred), file=out_file)

output = evaluate(x_peer, encoder, decoder)
predict = []
for y in output:
    seq = []
    for w in y:
        if len(seq) == 0 or w != seq[-1]:
            seq.append(w)
    seq = ' '.join([index_word[i] for i in seq]).strip()
    seq = seq[0].upper() + seq[1:]
    predict.append(seq)
with open(sys.argv[3], 'w+') as out_file:
    for id, pred in zip(peer_id, predict):
        print('{},{}'.format(id, pred), file=out_file)
