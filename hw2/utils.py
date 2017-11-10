import os
import json
import numpy as np
import pickle

def load_data(data_dir):
    train_feat_dir = os.path.join(data_dir, 'training_data/feat')
    train_label_path = os.path.join(data_dir, 'training_label.json')
    test_feat_dir = os.path.join(data_dir, 'testing_data/feat')
    test_label_path = os.path.join(data_dir, 'testing_label.json')

    train_list = os.listdir(train_feat_dir)
    test_list = os.listdir(test_feat_dir)
    
    feature = []
    for filename in train_list:
        feature.append(np.load(os.path.join(train_feat_dir, filename)))
    x_train = np.array(feature).astype(np.float32)

    label = json.load(open(train_label_path))
    label_dict = {}

    for x in label:
        label_dict[x['id'].strip() + '.npy'] = x['caption']

    max_seq = 0
    word_cnt = {}
    y_train = []
    
    for x, filename in zip(feature, train_list):
        y_train.append([])
        for caption in label_dict[filename]:
            y_train[-1].append(caption.strip('.').lower().split(' '))
            max_seq = max(max_seq, len(y_train[-1][-1]))
    '''
            for v in y_train[-1][-1]:
                try:
                    word_cnt[v] += 1
                except KeyError:
                    word_cnt[v] = 1

    word_index = {}
    index_word = ['']
    word_size = 0
    for v, c in word_cnt.items():
        if c > 0:
            word_index[v] = word_size + 1
            index_word.append(v)
            word_size += 1
    pickle.dump(word_index, open('wtoi.pkl', 'wb'))
    pickle.dump(index_word, open('itow.pkl', 'wb'))
    '''
    word_index = pickle.load(open('wtoi.pkl', 'rb'))
    y_train = [[[word_index[w] for w in seq] for seq in seqs] for seqs in y_train]
    y_train = [np.array([seq + [0] * (max_seq - len(seq)) for seq in seqs]).astype(np.int64) for seqs in y_train]
    y_train = np.array(y_train)
    
    feature = []
    for filename in test_list:
        feature.append(np.load(os.path.join(test_feat_dir, filename)))
    x_test = np.array(feature).astype(np.float32)
    
    label = json.load(open(test_label_path))
    label_dict = {}
    y_test = []
    test_id = []
    for x in label:
        label_dict[x['id'].strip() + '.npy'] = x['caption']
    for x, filename in zip(feature, test_list):
        y_test.append(label_dict[filename])
        test_id.append(filename[:-4])
    return (x_train, y_train), (x_test, y_test, test_id)

