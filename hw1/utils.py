import os
import numpy as np
import editdistance
import re
phones_char={'':'','aa':'a','ae':'b','ah':'c','aw':'e','ay':'g','b':'h','ch':'i','d':'k','dh':'l','dx':'m','eh':'n','er':'r','ey':'s','f':'t','g':'u','hh':'v','ih':'w','iy':'y','jh':'z','k':'A','l':'B','m':'C','n':'D','ng':'E','ow':'F','oy':'G','p':'H','r':'I','s':'J','sh':'K','sil':'L','t':'M','th':'N','uh':'O','uw':'P','v':'Q','w':'S','y':'T','z':'U'}
phones_48_39={'aa':'aa','ae':'ae','ah':'ah','ao':'aa','aw':'aw','ax':'ah','ay':'ay','b':'b','ch':'ch','cl':'sil','d':'d','dh':'dh','dx':'dx','eh':'eh','el':'l','en':'n','epi':'sil','er':'er','ey':'ey','f':'f','g':'g','hh':'hh','ih':'ih','ix':'ih','iy':'iy','jh':'jh','k':'k','l':'l','m':'m','ng':'ng','n':'n','ow':'ow','oy':'oy','p':'p','r':'r','sh':'sh','sil':'sil','s':'s','th':'th','t':'t','uh':'uh','uw':'uw','vcl':'sil','v':'v','w':'w','y':'y','zh':'sh','z':'z'}
char_index={'':0,'a':1,'b':2,'c':3,'e':4,'g':5,'h':6,'i':7,'k':8,'l':9,'m':10,'n':11,'r':12,'s':13,'t':14,'u':15,'v':16,'w':17,'y':18,'z':19,'A':20,'B':21,'C':22,'D':23,'E':24,'F':25,'G':26,'H':27,'I':28,'J':29,'K':30,'L':31,'M':32,'N':33,'O':34,'P':35,'Q':36,'S':37,'T':38,'U':39}
index_char=['','a','b','c','e','g','h','i','k','l','m','n','r','s','t','u','v','w','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','S','T','U']

def load_phones_map(): 
    phones_char = {'': ''}
    phones_48_39 = {}
    char_index = {}
    index_char = []

    with open('data/48phone_char.map', 'r') as file:
        for line in file:
            line = line.replace('\n', '').split('\t')
            phones_char[line[0]] = line[2]
    
    with open('data/phones/48_39.map', 'r') as file:
        for line in file:
            line = line.replace('\n', '').split('\t')
            phones_48_39[line[0]] = line[1]
            if line[0] != line[1]: 
                del phones_char[line[0]]
    
    for i, (phone, char) in enumerate(phones_char.items()):
        index_char.append(char)
        char_index[char] = i

    return phones_char, phones_48_39, char_index, index_char

#phones_char, phones_48_39, char_index, index_char = load_phones_map()

def encode(x):
    ret = []
    for i in x:
        seq = ''.join([index_char[c] for c in i])
        seq = re.sub(r'(.)\1+', r'\1', seq).strip('L')
        ret.append(seq)
    return ret

def avg_ed(x, y):
    ret = 0
    for i in range(len(x)):
        ret += editdistance.eval(x[i], y[i])
    return ret# / len(x)

def load_feature_data(path):
    with open(path) as feature_file:
        feature = []
        for line in feature_file:
            feature.append(line.replace('\n', '').split(' '))
            
            if line[0] == 'f':
                feature[-1].append(0)
                feature[-1].append(1)
            else:
                feature[-1].append(1)
                feature[-1].append(0)
            
    return feature

def load_label_data(path):
    with open(path) as label_file:
        label = []
        for line in label_file:
            label.append(line.replace('\n', '').split(','))
    return label

def load_test_data(dataset):
    path = os.path.join('data/', dataset, 'test.ark')
    feature = load_feature_data(path)
    
    max_seql = 777
    
    x_test = []
    test_id = []
    pre_id = '' 
    tmp = []
    for i, x in enumerate(feature):
        if i != 0 and x[0].split('_')[-1] == '1':
            x_test.append(tmp)
            tmp = []
        tmp.append(x[1:])
        seq_id = '_'.join(x[0].split('_')[:-1])
        if seq_id != pre_id:
            pre_id = seq_id
            test_id.append(seq_id)
    x_test.append(tmp)
    
    for x in x_test:
        x += [[0] * len(x[0])] * (max_seql - len(x))
    
    x_test = np.array(x_test).astype(np.float32)

    return x_test, test_id
                
def load_train_data(dataset):
   
    feature_path = os.path.join('data/', dataset, 'train.ark')
    label_path = os.path.join('data/label', 'train.lab')

    max_seql = 777
    
    feature = load_feature_data(feature_path)
    label = load_label_data(label_path)

    feature = sorted(feature, key=lambda x: x[0].split('_')[:-1])
    label = sorted(label, key=lambda y: y[0].split('_')[:-1])
    
    x_train = []
    y_train = []
    tmp = []
    
    for i, x in enumerate(feature):
        if i != 0 and x[0].split('_')[-1] == '1':
            x_train.append(tmp)
            tmp = []
        tmp.append(x[1:])
    x_train.append(tmp)

    tmp = []
    for i, y in enumerate(label):
        if i != 0 and y[0].split('_')[-1] =='1':
            y_train.append(tmp)
            tmp = []
        tmp.append(char_index[phones_char[phones_48_39[y[-1]]]])
    y_train.append(tmp)
    
    # Padding
    for x in x_train:
        x += [[0] * len(x[0])] * (max_seql - len(x))
    for y in y_train:
        y += [0] * (max_seql - len(y))

    x_train = np.array(x_train).astype(np.float32)
    y_train = np.array(y_train).astype(np.int64)

    return x_train, y_train

def split_validation(x_train, y_train, ratio):
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    
    x_train, y_train = x_train[idx], y_train[idx]
    
    n_val = int(x_train.shape[0] * ratio)

    x_valid, y_valid = x_train[:n_val], y_train[:n_val]
    x_train, y_train = x_train[n_val:], y_train[n_val:]

    return (x_train, y_train), (x_valid, y_valid)

