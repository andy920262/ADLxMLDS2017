import sys
import pickle
import numpy as np
from utils import *
from model import *
import torch
from torch.autograd import Variable

from data.bleu_eval import BLEU

data_dir = 'data'
epochs = 20
batch_size = 64

def train(model, optimizer, loss_func, x_train, y_train):
    x_train = Variable(torch.FloatTensor(x_train)).cuda()
    y_train = Variable(torch.LongTensor(y_train)).cuda()
    pred = model(x_train)
    loss = loss_func(pred, y_train.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.data[0] / x_train.size(0)


if __name__ == '__main__':
    print('Loading data.')
    #(x_train, y_train), (x_test, y_test, test_id) =  pickle.load(open('data.pkl', 'rb'))
    (x_train, y_train), (x_test, y_test, test_id) = load_data(data_dir)
    #pickle.dump(((x_train, y_train), (x_test, y_test, test_id)), open('data.pkl', 'wb')) 
    index_word = pickle.load(open('itow.pkl', 'rb'))

    input_size = x_train.shape[-1]
    output_size = len(index_word)

    print('Initial model.')

    model = S2VT(input_size, output_size).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()
   
    print('Start training')
    for epoch in range(1, epochs + 1):
        total_loss, steps = 0, 0
        for batch_i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[batch_i : min(batch_i + batch_size, x_train.shape[0])]
            y_batch = []
            for y in y_train[batch_i : min(batch_i + batch_size, x_train.shape[0])]:
                y_batch.append(y[epoch % y.shape[0]])
            y_batch = np.array(y_batch).astype(np.int64)
                
            total_loss += train(model, optimizer, loss_func, x_batch, y_batch)
            steps += 1
        print('Epoch: {}, loss: {:.4f}'.format(epoch, total_loss / steps))
        
        # Test
        output = model(Variable(torch.FloatTensor(x_test)).cuda())
        output = torch.max(output, -1)[1].view(100, -1).cpu().data.numpy()
        predict = []
        for y in output:
            seq = ' '.join([index_word[i] for i in y]).strip() + '.'
            seq = seq[0].upper() + seq[1:]
            predict.append(seq)
        print(predict[0])
        total_bleu = 0
        for p, y in zip(predict, y_test):
            y_bleu = 0
            for c in y:
                y_bleu += BLEU(c, p)
            total_bleu += y_bleu / len(y)
        print('AVG BLEU = {}'.format(total_bleu / 100))
        
    with open('model.pt', 'wb') as file:
        torch.save(model.state_dict(), file)
        
    

