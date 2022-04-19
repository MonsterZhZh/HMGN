import numpy as np
import torch


def create_adjacency_matrix(trees, n_nodes, levels):
    # [src, tgt]
    am = np.zeros([n_nodes, n_nodes])
    for tree in trees:
        # [specie, order, family]
        if levels == 2:
            am[tree[1]][tree[0]] = 1
            am[tree[0]][tree[1]] = 1
        elif levels == 3:
            am[tree[1]][tree[2]] = 1
            am[tree[2]][tree[1]] = 1
            am[tree[2]][tree[0]] = 1
            am[tree[0]][tree[2]] = 1
    return am


trees_CUB = [
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


trees_family_to_order_CUB = [
    [1, 1], 
    [2, 2], 
    [3, 3], 
    [4, 4], 
    [5, 4], 
    [6, 4], 
    [7, 5], 
    [8, 5], 
    [9, 6], 
    [10, 7], 
    [11, 8], 
    [12, 8], 
    [13, 8], 
    [14, 8], 
    [15, 8], 
    [16, 8], 
    [17, 8], 
    [18, 8], 
    [19, 8], 
    [20, 8], 
    [21, 8], 
    [22, 8], 
    [23, 8], 
    [24, 8], 
    [25, 8], 
    [26, 8], 
    [27, 8], 
    [28, 8], 
    [29, 8], 
    [30, 8], 
    [31, 8], 
    [32, 9], 
    [33, 10], 
    [34, 11], 
    [35, 12], 
    [36, 12], 
    [37, 13], 
    [38, 13]
]


trees_Air = [
    [1, 1, 1],
    [2, 2, 1],
    [3, 3, 1],
    [4, 3, 1],
    [5, 3, 1],
    [6, 3, 1],
    [7, 4, 1],
    [8, 4, 1],
    [9, 5, 1],
    [10, 5, 1],
    [11, 5, 1],
    [12, 5, 1],
    [13, 6, 1],
    [14, 7, 2],
    [15, 8, 3],
    [16, 9, 3],
    [17, 10, 7],
    [18, 10, 7],
    [19, 11, 7],
    [20, 12, 4],
    [21, 13, 5],
    [22, 14, 5],
    [23, 15, 5],
    [24, 16, 5],
    [25, 16, 5],
    [26, 16, 5],
    [27, 16, 5],
    [28, 16, 5],
    [29, 16, 5],
    [30, 16, 5],
    [31, 16, 5],
    [32, 17, 5],
    [33, 17, 5],
    [34, 17, 5],
    [35, 17, 5],
    [36, 18, 5],
    [37, 18, 5],
    [38, 19, 5],
    [39, 19, 5],
    [40, 19, 5],
    [41, 20, 5],
    [42, 20, 5],
    [43, 21, 21],
    [44, 22, 14],
    [45, 23, 9],
    [46, 24, 9],
    [47, 25, 9],
    [48, 25, 9],
    [49, 26, 8],
    [50, 27, 8],
    [51, 28, 8],
    [52, 28, 8],
    [53, 29, 12],
    [54, 29, 12],
    [55, 30, 23],
    [56, 31, 14],
    [57, 32, 14],
    [58, 33, 14],
    [59, 34, 23],
    [60, 35, 12],
    [61, 36, 12],
    [62, 37, 12],
    [63, 38, 13],
    [64, 39, 26],
    [65, 40, 15],
    [66, 41, 15],
    [67, 41, 15],
    [68, 41, 15],
    [69, 42, 15],
    [70, 42, 15],
    [71, 43, 15],
    [72, 44, 16],
    [73, 45, 23],
    [74, 46, 22],
    [75, 47, 11],
    [76, 48, 11],
    [77, 49, 18],
    [78, 50, 18],
    [79, 51, 18],
    [80, 52, 6],
    [81, 53, 19],
    [82, 53, 19],
    [83, 54, 7],
    [84, 55, 20],
    [85, 56, 4],
    [86, 57, 21],
    [87, 58, 23],
    [88, 59, 23],
    [89, 59, 23],
    [90, 60, 23],
    [91, 61, 17],
    [92, 62, 25],
    [93, 63, 27],
    [94, 64, 27],
    [95, 65, 28],
    [96, 66, 10],
    [97, 67, 24],
    [98, 68, 29],
    [99, 69, 29],
    [100, 70, 30]
]


trees_family_to_order_Air = [
    [1, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 1],
    [6, 1],
    [7, 2],
    [8, 3],
    [9, 3],
    [12, 4],
    [56, 4],
    [13, 5],
    [14, 5],
    [15, 5],
    [16, 5],
    [17, 5],
    [18, 5],
    [19, 5],
    [20, 5],
    [52, 6],
    [10, 7],
    [11, 7],
    [54, 7],
    [26, 8],
    [27, 8],
    [28, 8],
    [23, 9],
    [24, 9],
    [25, 9],
    [66, 10],
    [47, 11],
    [48, 11],
    [29, 12],
    [35, 12],
    [36, 12],
    [37, 12],
    [38, 13],
    [22, 14],
    [31, 14],
    [32, 14],
    [33, 14],
    [40, 15],
    [41, 15],
    [42, 15],
    [43, 15],
    [44, 16],
    [61, 17],
    [49, 18],
    [50, 18],
    [51, 18],
    [53, 19],
    [55, 20],
    [21, 21],
    [57, 21],
    [46, 22],
    [30, 23],
    [34, 23],
    [45, 23],
    [58, 23],
    [59, 23],
    [60, 23],
    [67, 24],
    [62, 25],
    [39, 26],
    [63, 27],
    [64, 27],
    [65, 28],
    [68, 29],
    [69, 29],
    [70, 30]
]


def get_order_family_target(targets, dataset, device):
    '''
    把一个batch中的每个样本所属的order label添加到order_target_list，值为0~12
    把一个batch中的每个样本所属的family label添加到family_target_list，值为13~50
    '''

    order_target_list = []
    family_target_list = []
    target_list_sig = []

    for i in range(targets.size(0)):
        if dataset == 'CUB':
            if targets[i] < 51 and targets[i] > 12:   # 即选出属于family的标签
                order_target_list.append(trees_family_to_order_CUB[targets[i]-13][1]-1)
                family_target_list.append(int(targets[i]-13))
            elif targets[i] > 50:
                order_target_list.append(trees_CUB[targets[i]-51][1]-1)
                family_target_list.append(trees_CUB[targets[i]-51][2]-1)

        elif dataset == 'Air':
            if targets[i] < 100 and targets[i] > 29:   # 即选出属于family的标签
                order_target_list.append(trees_family_to_order_Air[targets[i]-30][1]-1)
                family_target_list.append(int(targets[i]-30))
            elif targets[i] > 99:
                order_target_list.append(trees_Air[targets[i]-100][2]-1)
                family_target_list.append(trees_Air[targets[i]-100][1]-1)
    
        target_list_sig.append(int(targets[i]))
    
    order_target_list = torch.from_numpy(np.array(order_target_list)).to(device)
    family_target_list = torch.from_numpy(np.array(family_target_list)).to(device)
    target_list_sig = torch.from_numpy(np.array(target_list_sig)).to(device)
    return order_target_list, family_target_list, target_list_sig


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)