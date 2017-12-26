import os
import skimage
import skimage.io
import skimage.transform
import numpy as np
import random

itoc = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
    'green hair', 'red hair', 'purple hair', 'pink hair',
    'blue hair', 'black hair', 'brown hair', 'blonde hair', '<unk> hair',
    'gray eyes', 'black eyes', 'orange eyes',
    'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
    'green eyes', 'brown eyes', 'red eyes', 'blue eyes', '<unk> eyes']
ctoi = {'orange hair': 0, 'white hair': 1, 'aqua hair': 2, 'gray hair': 3,
    'green hair': 4, 'red hair': 5, 'purple hair': 6, 'pink hair': 7,
    'blue hair': 8, 'black hair': 9, 'brown hair': 10, 'blonde hair': 11,
    '<unk> hair': 12, 'gray eyes': 13, 'black eyes': 14, 'orange eyes': 15,
    'pink eyes': 16, 'yellow eyes': 17, 'aqua eyes': 18, 'purple eyes': 19,
    'green eyes': 20, 'brown eyes': 21, 'red eyes': 22, 'blue eyes': 23, '<unk> eyes': 24}

def tag_one_hot(line):
    tag = [0 if line.find(c) == -1 else 1 for c in itoc]
    if sum(tag[:12]) == 0:
        tag[12] = 1
    if sum(y_train[-1][13:]) == 0:
        tag[24] = 1
    return tag

def load_train_data(path):
    
    x_train, y_train = [], []
    
    for index in range(33431):
        filename = '{}.jpg'.format(index)
        img = skimage.io.imread(os.path.join(path, 'faces', filename))
        img = skimage.transform.resize(img, (64, 64))
        x_train.append(img)
    
    with open(os.path.join(path, 'tags_clean.csv')) as fd:
        for line in fd:
            y.append(tag_one_hot(line))
    
    x_train = np.array(x_train)
    y_train = np.array(y_train).astype(np.float64)

    return x_train, y_train

if __name__ == '__main__':
    x_train, y_train = load_train_data('data')
