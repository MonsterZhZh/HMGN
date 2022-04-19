import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms

from os.path import join
from PIL import Image
import random
import math
import os
import networkx as nx
import numpy as np


class CubDataset(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform=None, re_level='class', proportion=1.0):
        super(CubDataset, self).__init__()

        self.re_level = re_level
        self.proportion = proportion
        self.trees = [
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

        name_list = []
        family_label_list = []
        species_label_list = []

        with open(list_path, 'r') as f:
            for l in f.readlines():
                imagename, class_label, genus_label, family_label, order_label = l.strip().split(' ')
                name_list.append(imagename)
                family_label_list.append(self.trees[int(class_label)-1][-1] + 13)
                species_label_list.append(int(class_label) + 51)

        image_filenames = [join(image_dir, x) for x in name_list]
        self.input_transform = input_transform

        self.image_filenames, self.labels = self.relabel(image_filenames, family_label_list, species_label_list)

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        target = self.labels[index] - 1

        return input, target

    def __len__(self):
        return len(self.image_filenames)

    def relabel(self, image_filenames, family_label_list, species_label_list):
        class_imgs = {}
        for i in range(len(image_filenames)):
            if str(species_label_list[i]) not in class_imgs.keys():
                class_imgs[str(species_label_list[i])] = {'images': [], 'family': []}
                class_imgs[str(species_label_list[i])]['images'].append(image_filenames[i])
                class_imgs[str(species_label_list[i])]['family'].append(family_label_list[i])
            else:
                class_imgs[str(species_label_list[i])]['images'].append(image_filenames[i])
        labels = []
        images = []
        for key in class_imgs.keys():
            # random.shuffle(class_imgs[key]['images'])
            images += class_imgs[key]['images']
            labels += [int(key)] * math.ceil(len(class_imgs[key]['images']) * self.proportion)
            rest = len(class_imgs[key]['images']) - math.ceil(len(class_imgs[key]['images']) * self.proportion)
            # print(key + ' has the rest: ' + str(rest))
            if self.re_level == 'family':
                labels += class_imgs[key]['family'] * rest
            elif self.re_level == 'class':
                labels += [int(key)] * rest
            else:
                print('Unrecognized level!!!')

        return images, labels



class AirDataset(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform=None, re_level='class', proportion=1.0):
        super(AirDataset, self).__init__()

        self.re_level = re_level
        self.proportion = proportion
        self.trees = [
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
        self.map = {'A300B4': 1, 'A310': 2, 'A318': 3, 'A319': 4, 'A320': 5, 'A321': 6, 'A330-200': 7, 'A330-300': 8, 'A340-200': 9, 'A340-300': 10, 'A340-500': 11, 'A340-600': 12, 'A380': 13, 'An-12': 14, 'ATR-42': 15, 'ATR-72': 16, 'BAE 146-200': 17, 'BAE 146-300': 18, 'BAE-125': 19, 'Beechcraft 1900': 20, '707-320': 21, 'Boeing 717': 22, '727-200': 23, '737-200': 24, '737-300': 25, '737-400': 26, '737-500': 27, '737-600': 28, '737-700': 29, '737-800': 30, '737-900': 31, '747-100': 32, '747-200': 33, '747-300': 34, '747-400': 35, '757-200': 36, '757-300': 37, '767-200': 38, '767-300': 39, '767-400': 40, '777-200': 41, '777-300': 42, 'C-130': 43, 'C-47': 44, 'Cessna 172': 45, 'Cessna 208': 46, 'Cessna 525': 47, 'Cessna 560': 48, 'Challenger 600': 49, 'CRJ-200': 50, 'CRJ-700': 51, 'CRJ-900': 52, 'DHC-8-100': 53, 'DHC-8-300': 54, 'DC-10': 55, 'DC-3': 56, 'DC-6': 57, 'DC-8': 58, 'DC-9-30': 59, 'DH-82': 60, 'DHC-1': 61, 'DHC-6': 62, 'Dornier 328': 63, 'DR-400': 64, 'EMB-120': 65, 'E-170': 66, 'E-190': 67, 'E-195': 68, 'ERJ 135': 69, 'ERJ 145': 70, 'Embraer Legacy 600': 71, 'Eurofighter Typhoon': 72, 'F/A-18': 73, 'F-16A/B': 74, 'Falcon 2000': 75, 'Falcon 900': 76, 'Fokker 100': 77, 'Fokker 50': 78, 'Fokker 70': 79, 'Global Express': 80, 'Gulfstream IV': 81, 'Gulfstream V': 82, 'Hawk T1': 83, 'Il-76': 84, 'Model B200': 85, 'L-1011': 86, 'MD-11': 87, 'MD-80': 88, 'MD-87': 89, 'MD-90': 90, 'Metroliner': 91, 'PA-28': 92, 'Saab 2000': 93, 'Saab 340': 94, 'Spitfire': 95, 'SR-20': 96, 'Tornado': 97, 'Tu-134': 98, 'Tu-154': 99, 'Yak-42': 100}

        name_list = []
        family_label_list = []
        species_label_list = []

        with open(list_path, 'r') as f:
            for l in f.readlines():
                lists = l.strip().strip('\n').split(' ')
                imagename = lists[0]
                classname = " ".join(i for i in lists[1:])
                name_list.append(imagename + '.jpg')
                class_label = self.map[classname]
                family_label_list.append(self.trees[class_label-1][1] + 30)
                species_label_list.append(class_label + 100)

        image_filenames = [join(image_dir, x) for x in name_list]
        self.input_transform = input_transform

        self.image_filenames, self.labels = self.relabel(image_filenames, family_label_list, species_label_list)

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        target = self.labels[index] - 1

        return input, target

    def __len__(self):
        return len(self.image_filenames)

    def relabel(self, image_filenames, family_label_list, species_label_list):
        class_imgs = {}
        for i in range(len(image_filenames)):
            if str(species_label_list[i]) not in class_imgs.keys():
                class_imgs[str(species_label_list[i])] = {'images': [], 'family': []}
                class_imgs[str(species_label_list[i])]['images'].append(image_filenames[i])
                class_imgs[str(species_label_list[i])]['family'].append(family_label_list[i])
            else:
                class_imgs[str(species_label_list[i])]['images'].append(image_filenames[i])
        labels = []
        images = []
        for key in class_imgs.keys():
            # random.shuffle(class_imgs[key]['images'])
            images += class_imgs[key]['images']
            labels += [int(key)] * math.ceil(len(class_imgs[key]['images']) * self.proportion)
            rest = len(class_imgs[key]['images']) - math.ceil(len(class_imgs[key]['images']) * self.proportion)
            # print(key + ' has the rest: ' + str(rest))
            if self.re_level == 'family':
                labels += class_imgs[key]['family'] * rest
            elif self.re_level == 'class':
                labels += [int(key)] * rest
            else:
                print('Unrecognized level!!!')

        return images, labels



class CarDataset(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform=None, re_level='class', proportion=1.0):
        super(CarDataset, self).__init__()

        self.re_level = re_level
        self.proportion = proportion
        self.trees = [
            [1, 7],
            [2, 6],
            [3, 6],
            [4, 6],
            [5, 6],
            [6, 3],
            [7, 4],
            [8, 2],
            [9, 3],
            [10, 2],
            [11, 3],
            [12, 2],
            [13, 3],
            [14, 3],
            [15, 3],
            [16, 6],
            [17, 6],
            [18, 9],
            [19, 4],
            [20, 6],
            [21, 2],
            [22, 3],
            [23, 6],
            [24, 6],
            [25, 3],
            [26, 6],
            [27, 2],
            [28, 3],
            [29, 6],
            [30, 9],
            [31, 2],
            [32, 7],
            [33, 7],
            [34, 3],
            [35, 6],
            [36, 2],
            [37, 7],
            [38, 2],
            [39, 2],
            [40, 6],
            [41, 6],
            [42, 3],
            [43, 3],
            [44, 6],
            [45, 2],
            [46, 3],
            [47, 6],
            [48, 7],
            [49, 6],
            [50, 7],
            [51, 6],
            [52, 7],
            [53, 1],
            [54, 1],
            [55, 2],
            [56, 3],
            [57, 3],
            [58, 7],
            [59, 2],
            [60, 5],
            [61, 6],
            [62, 7],
            [63, 6],
            [64, 8],
            [65, 1],
            [66, 3],
            [67, 6],
            [68, 7],
            [69, 1],
            [70, 1],
            [71, 8],
            [72, 3],
            [73, 6],
            [74, 1],
            [75, 1],
            [76, 7],
            [77, 2],
            [78, 5],
            [79, 6],
            [80, 2],
            [81, 2],
            [82, 9],
            [83, 9],
            [84, 9],
            [85, 5],
            [86, 1],
            [87, 1],
            [88, 8],
            [89, 7],
            [90, 1],
            [91, 1],
            [92, 9],
            [93, 3],
            [94, 7],
            [95, 7],
            [96, 6],
            [97, 6],
            [98, 4],
            [99, 3],
            [100, 2],
            [101, 3],
            [102, 2],
            [103, 2],
            [104, 3],
            [105, 6],
            [106, 1],
            [107, 2],
            [108, 5],
            [109, 7],
            [110, 7],
            [111, 1],
            [112, 3],
            [113, 1],
            [114, 1],
            [115, 6],
            [116, 9],
            [117, 6],
            [118, 7],
            [119, 8],
            [120, 7],
            [121, 7],
            [122, 1],
            [123, 2],
            [124, 1],
            [125, 1],
            [126, 5],
            [127, 5],
            [128, 3],
            [129, 6],
            [130, 4],
            [131, 7],
            [132, 7],
            [133, 7],
            [134, 6],
            [135, 6],
            [136, 6],
            [137, 6],
            [138, 6],
            [139, 4],
            [140, 6],
            [141, 3],
            [142, 7],
            [143, 7],
            [144, 3],
            [145, 7],
            [146, 7],
            [147, 7],
            [148, 7],
            [149, 7],
            [150, 3],
            [151, 3],
            [152, 3],
            [153, 3],
            [154, 7],
            [155, 7],
            [156, 6],
            [157, 2],
            [158, 2],
            [159, 7],
            [160, 3],
            [161, 2],
            [162, 6],
            [163, 3],
            [164, 6],
            [165, 6],
            [166, 8],
            [167, 6],
            [168, 4],
            [169, 8],
            [170, 4],
            [171, 3],
            [172, 3],
            [173, 6],
            [174, 5],
            [175, 2],
            [176, 6],
            [177, 6],
            [178, 4],
            [179, 2],
            [180, 3],
            [181, 6],
            [182, 6],
            [183, 4],
            [184, 6],
            [185, 6],
            [186, 7],
            [187, 6],
            [188, 6],
            [189, 7],
            [190, 4],
            [191, 4],
            [192, 4],
            [193, 4],
            [194, 6],
            [195, 7],
            [196, 2]
        ]

        name_list = []
        family_label_list = []
        species_label_list = []

        with open(list_path, 'r') as f:
            for l in f.readlines():
                imagename, classname = l.strip().strip('\n').split(' ')
                name_list.append(imagename)
                class_label = int(classname)
                family_label_list.append(self.trees[class_label-1][1])
                species_label_list.append(class_label + 9)

        image_filenames = [join(image_dir, x) for x in name_list]
        self.input_transform = input_transform

        self.image_filenames, self.labels = self.relabel(image_filenames, family_label_list, species_label_list)

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        target = self.labels[index] - 1

        return input, target

    def __len__(self):
        return len(self.image_filenames)

    def relabel(self, image_filenames, family_label_list, species_label_list):
        class_imgs = {}
        for i in range(len(image_filenames)):
            if str(species_label_list[i]) not in class_imgs.keys():
                class_imgs[str(species_label_list[i])] = {'images': [], 'family': []}
                class_imgs[str(species_label_list[i])]['images'].append(image_filenames[i])
                class_imgs[str(species_label_list[i])]['family'].append(family_label_list[i])
            else:
                class_imgs[str(species_label_list[i])]['images'].append(image_filenames[i])
        labels = []
        images = []
        for key in class_imgs.keys():
            # random.shuffle(class_imgs[key]['images'])
            images += class_imgs[key]['images']
            labels += [int(key)] * math.ceil(len(class_imgs[key]['images']) * self.proportion)
            rest = len(class_imgs[key]['images']) - math.ceil(len(class_imgs[key]['images']) * self.proportion)
            # print(key + ' has the rest: ' + str(rest))
            if self.re_level == 'family':
                labels += class_imgs[key]['family'] * rest
            elif self.re_level == 'class':
                labels += [int(key)] * rest
            else:
                print('Unrecognized level!!!')

        return images, labels