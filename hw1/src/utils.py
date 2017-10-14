import os
import numpy as np

def load_phones_map():
    
    phones_map = {}

    with open('data/48phone_char.map', 'r') as file:
        for line in file:
            line = line.replace('\n', '').split('\t')
            phones_map[line[0]] = [int(line[1]), line[2]]
    
    with open('data/phones/48_39.map', 'r') as file:
        for line in file:
            line = line.replace('\n', '').split('\t')
            phones_map[line[0]][1] = phones_map[line[1]][1]

    return phones_map

def load_train_data(dataset):
    return np.zeros((1000, 800, 39)), np.zeros((1000, 800, 49)) 
    feature_path = os.path.join('data/', dataset, 'train.ark')
    label_path = os.path.join('data/', 'label/', 'train.lab')

    phones_map = load_phones_map()

    max_seql = 800

    with open(feature_path) as feature_file:
        feature = []
        for line in feature_file:
            feature.append(line.replace('\n', '').split(' '))

    with open(label_path) as label_file:
        label = []
        for line in label_file:
            label.append(line.replace('\n', '').split(','))
    
    # Align by ID
    feature = sorted(feature)
    label = sorted(label)
    
    x_train = []
    y_train = []
    tmp = []

    for i, x in enumerate(feature):
        if i != 0 and x[0].split('_')[-1] =='1':
            tmp = tmp + [[0] * (len(x) - 1)] * (max_seql - len(tmp))
            x_train.append(tmp)
            tmp = []
        tmp.append(x[1:])
    tmp = tmp + [[0] * (len(x) - 1)] * (max_seql - len(tmp))
    x_train.append(tmp)

    
    tmp = []
    for i, y in enumerate(label):
        if i != 0 and y[0].split('_')[-1] =='1':
            tmp = tmp + [[0] * 48 + [1]] * (max_seql - len(tmp))
            y_train.append(tmp)
            tmp = []
        tmp.append([1 if phones_map[y[-1]][0] == c else 0 for c in range(49)])
    tmp = tmp + [[0] * 48 + [1]] * (max_seql - len(tmp))
    y_train.append(tmp)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train

def split_validation(x_train, y_train, ratio):
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    
    x_train, y_train = x_train[idx], y_train[idx]
    
    n_val = int(x_train.shape[0] * ratio)

    x_valid, y_valid = x_train[:n_val], y_train[:n_val]
    x_train, y_train = x_train[n_val:], y_train[n_val:]

    return (x_train, y_train), (x_valid, y_valid)

