'''
Reference:
https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
'''


import bcolz
import numpy as np
import pickle



CUB_trees = [
    [1,12,35],
    [2,12,35],
    [3,12,35],
    [4,6,9],
    [5,4,4],
    [6,4,4],
    [7,4,4],
    [8,4,4],
    [9,8,18],
    [10,8,18],
    [11,8,18],
    [12,8,18],
    [13,8,18],
    [14,8,13],
    [15,8,13],
    [16,8,13],
    [17,8,13],
    [18,8,26],
    [19,8,21],
    [20,8,19],
    [21,8,24],
    [22,3,3],
    [23,13,37],
    [24,13,37],
    [25,13,37],
    [26,8,18],
    [27,8,18],
    [28,8,14],
    [29,8,15],
    [30,8,15],
    [31,6,9],
    [32,6,9],
    [33,6,9],
    [34,8,16],
    [35,8,16],
    [36,10,33],
    [37,8,30],
    [38,8,30],
    [39,8,30],
    [40,8,30],
    [41,8,30],
    [42,8,30],
    [43,8,30],
    [44,13,38],
    [45,12,36],
    [46,1,1],
    [47,8,16],
    [48,8,16],
    [49,8,18],
    [50,11,34],
    [51,11,34],
    [52,11,34],
    [53,11,34],
    [54,8,13],
    [55,8,16],
    [56,8,16],
    [57,8,13],
    [58,4,4],
    [59,4,5],
    [60,4,5],
    [61,4,5],
    [62,4,5],
    [63,4,5],
    [64,4,5],
    [65,4,5],
    [66,4,5],
    [67,2,2],
    [68,2,2],
    [69,2,2],
    [70,2,2],
    [71,4,6],
    [72,4,6],
    [73,8,15],
    [74,8,15],
    [75,8,15],
    [76,8,24],
    [77,8,30],
    [78,8,30],
    [79,5,7],
    [80,5,7],
    [81,5,7],
    [82,5,7],
    [83,5,7],
    [84,5,8],
    [85,8,11],
    [86,7,10],
    [87,1,1],
    [88,8,18],
    [89,1,1],
    [90,1,1],
    [91,8,21],
    [92,3,3],
    [93,8,15],
    [94,8,27],
    [95,8,18],
    [96,8,18],
    [97,8,18],
    [98,8,18],
    [99,8,23],
    [100,9,32],
    [101,9,32],
    [102,8,30],
    [103,8,30],
    [104,8,22],
    [105,3,3],
    [106,4,4],
    [107,8,15],
    [108,8,15],
    [109,8,23],
    [110,6,9],
    [111,8,20],
    [112,8,20],
    [113,8,24],
    [114,8,24],
    [115,8,24],
    [116,8,24],
    [117,8,24],
    [118,8,25],
    [119,8,24],
    [120,8,24],
    [121,8,24],
    [122,8,24],
    [123,8,24],
    [124,8,24],
    [125,8,24],
    [126,8,24],
    [127,8,24],
    [128,8,24],
    [129,8,24],
    [130,8,24],
    [131,8,24],
    [132,8,24],
    [133,8,24],
    [134,8,28],
    [135,8,17],
    [136,8,17],
    [137,8,17],
    [138,8,17],
    [139,8,13],
    [140,8,13],
    [141,4,5],
    [142,4,5],
    [143,4,5],
    [144,4,5],
    [145,4,5],
    [146,4,5],
    [147,4,5],
    [148,8,24],
    [149,8,21],
    [150,8,21],
    [151,8,31],
    [152,8,31],
    [153,8,31],
    [154,8,31],
    [155,8,31],
    [156,8,31],
    [157,8,31],
    [158,8,23],
    [159,8,23],
    [160,8,23],
    [161,8,23],
    [162,8,23],
    [163,8,23],
    [164,8,23],
    [165,8,23],
    [166,8,23],
    [167,8,23],
    [168,8,23],
    [169,8,23],
    [170,8,23],
    [171,8,23],
    [172,8,23],
    [173,8,23],
    [174,8,23],
    [175,8,23],
    [176,8,23],
    [177,8,23],
    [178,8,23],
    [179,8,23],
    [180,8,23],
    [181,8,23],
    [182,8,23],
    [183,8,23],
    [184,8,23],
    [185,8,12],
    [186,8,12],
    [187,10,33],
    [188,10,33],
    [189,10,33],
    [190,10,33],
    [191,10,33],
    [192,10,33],
    [193,8,29],
    [194,8,29],
    [195,8,29],
    [196,8,29],
    [197,8,29],
    [198,8,29],
    [199,8,29],
    [200,8,23]
]


def decode_hierarchy_word(trees, word_file, levels=3):
    order = {}
    species = {}
    if levels == 3:
        families = {}
    with open(word_file, 'rb') as f:
        for l in f:
            line = l.decode().strip().strip('\n').split( )
            if len(line) != 4:
                print(line)
            else:
                class_num = int(line[0])
                class_name = line[1].split('.')[1]
                order_num = trees[class_num-1][1]
                order_name = line[3]
                family_num = trees[class_num-1][2]
                family_name = line[2]

                if order_num not in order.keys():
                    order[order_num] = set()
                order[order_num].add(order_name)
                if family_num not in families.keys():
                    families[family_num] = set()
                families[family_num].add(family_name)
                
                species[class_num] = class_name
    return order, families, species



def glove_word_vectors(glove_path):
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.300.dat', mode='w')

    with open(f'{glove_path}/glove.6B.300d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
        
    vectors = bcolz.carray(vectors[1:].reshape((400000, 300)), rootdir=f'{glove_path}/6B.300.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/6B.300_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.300_idx.pkl', 'wb'))


def glove_word_embeddings(glove_path, word_file, trees, skip_words=['and'], total_nodes = 251):
    vectors = bcolz.open(f'{glove_path}/6B.300.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.300_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.300_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}
    word_embeddings = np.zeros((total_nodes, 300), float)

    order, families, species = decode_hierarchy_word(CUB_trees, word_file)

    for key in order.keys():
        embeddings = np.zeros(300, float)
        for word in order[key]:
            embeddings += glove[word.lower()]
        we = embeddings / len(order[key])
        word_embeddings[key-1, :] = we
    
    for key in families.keys():
        embeddings = np.zeros(300, float)
        for word in families[key]:
            embeddings += glove[word.lower()]
        we = embeddings / len(families[key])
        word_embeddings[key+12, :] = we

    for key in species.keys():
        embeddings = np.zeros(300, float)
        words = species[key].split('_')
        nums = len(words)
        for word in words:
            if word in skip_words:
                nums -= 1
                continue
            embeddings += glove[word.lower()]
        we = embeddings / nums
        word_embeddings[key+50, :] = we

    pickle.dump(word_embeddings, open(f'{glove_path}/word_embeddings_CUB.pkl', 'wb'))


if __name__ == '__main__':
    glove_path = './pre-trained/'
    word_file = '/home/datasets/HI_Datasets/CUB2011/CUB_200_2011/hierarchy_classes.txt'

    # Process word vectors
    # glove_word_vectors(glove_path)

    # Generate dataset related word embeddings
    # decode_hierarchy_word(CUB_trees, word_file)
    glove_word_embeddings(glove_path, word_file, CUB_trees)