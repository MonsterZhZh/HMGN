import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,2'

from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms, models
import torch.hub
import argparse
from torch.optim import lr_scheduler

from HMGN import HMGN

from tree_loss import TreeLoss
from dataset import CubDataset, AirDataset, CarDataset
# from autoaugment import AutoAugImageNetPolicy
from train_test import test
from train_test import train
from utils import create_adjacency_matrix



def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch Deployment')
    parser.add_argument('--balance', default=1.0, type=float, help='balance between ce and tree (default: 1.0)')
    parser.add_argument('--worker', default=4, type=int, help='number of workers (default: 4)')
    parser.add_argument('--model', type=str, default='./pre-trained/resnet50-19c8e357.pth', help='Path of pre-trained model')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--mode', type=int, default=0, help='HMC Test Modes (default: 0)')

    parser.add_argument('--proportion', type=float, default=1.0, help='Proportion of species label')  
    parser.add_argument('--epoch', default=200, type=int, help='Epochs (default: 60)')
    parser.add_argument('--batch', type=int, default=8, help='batch size (default: 32)')      
    parser.add_argument('--dataset', type=str, default='CUB', help='dataset name', choices=['CUB', 'Air', 'Car'])
    parser.add_argument('--img_size', type=str, default='448', help='dataset name')
    parser.add_argument('--lr_adjt', type=str, default='Cos', help='Learning rate schedual', choices=['Cos', 'Step'])
    parser.add_argument('--device', nargs='+', default='6', help='GPU IDs for DP training')   # , required=True

    args = parser.parse_args()

    if args.proportion == 0.1: 
        args.epoch = 100
        args.batch = 32
        args.lr_adjt = 'Step'
    
    return args


if __name__ == '__main__':
    args = arg_parse()
    print('==> proportion: ', args.proportion)
    print('==> epoch: ', args.epoch)
    print('==> batch: ', args.batch)
    print('==> dataset: ', args.dataset)
    print('==> img_size: ', args.img_size)
    print('==> device: ', args.device)
    print('==> lr_adjt: ', args.lr_adjt)

    # Hyper-parameters
    nb_epoch = args.epoch
    batch_size = args.batch
    num_workers = args.worker

    # Preprocess
    if args.img_size == '448':
        transform_train = transforms.Compose([
        # transforms.Resize((600, 600), Image.BILINEAR),
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        # transforms.RandomCrop((448, 448)),
        transforms.RandomHorizontalFlip(),
        # AutoAugImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            # transforms.Resize((600, 600), Image.BILINEAR),
            transforms.Resize((550, 550)),
            # transforms.CenterCrop((448, 448)),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif args.img_size == '224':
        transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    # Data
    # print('==> Preparing data..')
    if args.dataset == 'CUB':
        # CUB-200-2011
        data_dir = '/home/datasets/HI_Datasets/CUB2011/CUB_200_2011/images'
        train_list = '/home/datasets/HI_Datasets/CUB2011/CUB_200_2011/hierarchy/train_images_4_level_V1.txt'
        test_list = '/home/datasets/HI_Datasets/CUB2011/CUB_200_2011/hierarchy/test_images_4_level_V1.txt'
        trees = [
                [51, 11, 47],
                [52, 11, 47],
                [53, 11, 47],
                [54, 5, 21],
                [55, 3, 16],
                [56, 3, 16],
                [57, 3, 16],
                [58, 3, 16],
                [59, 7, 30],
                [60, 7, 30],
                [61, 7, 30],
                [62, 7, 30],
                [63, 7, 30],
                [64, 7, 25],
                [65, 7, 25],
                [66, 7, 25],
                [67, 7, 25],
                [68, 7, 38],
                [69, 7, 33],
                [70, 7, 31],
                [71, 7, 36],
                [72, 2, 15],
                [73, 12, 49],
                [74, 12, 49],
                [75, 12, 49],
                [76, 7, 30],
                [77, 7, 30],
                [78, 7, 26],
                [79, 7, 27],
                [80, 7, 27],
                [81, 5, 21],
                [82, 5, 21],
                [83, 5, 21],
                [84, 7, 28],
                [85, 7, 28],
                [86, 9, 45],
                [87, 7, 42],
                [88, 7, 42],
                [89, 7, 42],
                [90, 7, 42],
                [91, 7, 42],
                [92, 7, 42],
                [93, 7, 42],
                [94, 12, 50],
                [95, 11, 48],
                [96, 0, 13],
                [97, 7, 28],
                [98, 7, 28],
                [99, 7, 30],
                [100, 10, 46],
                [101, 10, 46],
                [102, 10, 46],
                [103, 10, 46],
                [104, 7, 25],
                [105, 7, 28],
                [106, 7, 28],
                [107, 7, 25],
                [108, 3, 16],
                [109, 3, 17],
                [110, 3, 17],
                [111, 3, 17],
                [112, 3, 17],
                [113, 3, 17],
                [114, 3, 17],
                [115, 3, 17],
                [116, 3, 17],
                [117, 1, 14],
                [118, 1, 14],
                [119, 1, 14],
                [120, 1, 14],
                [121, 3, 18],
                [122, 3, 18],
                [123, 7, 27],
                [124, 7, 27],
                [125, 7, 27],
                [126, 7, 36],
                [127, 7, 42],
                [128, 7, 42],
                [129, 4, 19],
                [130, 4, 19],
                [131, 4, 19],
                [132, 4, 19],
                [133, 4, 19],
                [134, 4, 20],
                [135, 7, 23],
                [136, 6, 22],
                [137, 0, 13],
                [138, 7, 30],
                [139, 0, 13],
                [140, 0, 13],
                [141, 7, 33],
                [142, 2, 15],
                [143, 7, 27],
                [144, 7, 39],
                [145, 7, 30],
                [146, 7, 30],
                [147, 7, 30],
                [148, 7, 30],
                [149, 7, 35],
                [150, 8, 44],
                [151, 8, 44],
                [152, 7, 42],
                [153, 7, 42],
                [154, 7, 34],
                [155, 2, 15],
                [156, 3, 16],
                [157, 7, 27],
                [158, 7, 27],
                [159, 7, 35],
                [160, 5, 21],
                [161, 7, 32],
                [162, 7, 32],
                [163, 7, 36],
                [164, 7, 36],
                [165, 7, 36],
                [166, 7, 36],
                [167, 7, 36],
                [168, 7, 37],
                [169, 7, 36],
                [170, 7, 36],
                [171, 7, 36],
                [172, 7, 36],
                [173, 7, 36],
                [174, 7, 36],
                [175, 7, 36],
                [176, 7, 36],
                [177, 7, 36],
                [178, 7, 36],
                [179, 7, 36],
                [180, 7, 36],
                [181, 7, 36],
                [182, 7, 36],
                [183, 7, 36],
                [184, 7, 40],
                [185, 7, 29],
                [186, 7, 29],
                [187, 7, 29],
                [188, 7, 29],
                [189, 7, 25],
                [190, 7, 25],
                [191, 3, 17],
                [192, 3, 17],
                [193, 3, 17],
                [194, 3, 17],
                [195, 3, 17],
                [196, 3, 17],
                [197, 3, 17],
                [198, 7, 36],
                [199, 7, 33],
                [200, 7, 33],
                [201, 7, 43],
                [202, 7, 43],
                [203, 7, 43],
                [204, 7, 43],
                [205, 7, 43],
                [206, 7, 43],
                [207, 7, 43],
                [208, 7, 35],
                [209, 7, 35],
                [210, 7, 35],
                [211, 7, 35],
                [212, 7, 35],
                [213, 7, 35],
                [214, 7, 35],
                [215, 7, 35],
                [216, 7, 35],
                [217, 7, 35],
                [218, 7, 35],
                [219, 7, 35],
                [220, 7, 35],
                [221, 7, 35],
                [222, 7, 35],
                [223, 7, 35],
                [224, 7, 35],
                [225, 7, 35],
                [226, 7, 35],
                [227, 7, 35],
                [228, 7, 35],
                [229, 7, 35],
                [230, 7, 35],
                [231, 7, 35],
                [232, 7, 35],
                [233, 7, 35],
                [234, 7, 35],
                [235, 7, 24],
                [236, 7, 24],
                [237, 9, 45],
                [238, 9, 45],
                [239, 9, 45],
                [240, 9, 45],
                [241, 9, 45],
                [242, 9, 45],
                [243, 7, 41],
                [244, 7, 41],
                [245, 7, 41],
                [246, 7, 41],
                [247, 7, 41],
                [248, 7, 41],
                [249, 7, 41],
                [250, 7, 35]
                ]
        levels = 3
        total_nodes = 251
        trainset = CubDataset(data_dir, train_list, transform_train, re_level='family', proportion=args.proportion)
        testset = CubDataset(data_dir, test_list, transform_test, re_level='class', proportion=1.0)
    elif args.dataset == 'Air':
        # Aircraft
        data_dir = '/home/datasets/HI_Datasets/Aircraft/fgvc-aircraft-2013b/data/images'
        train_list = '/home/datasets/HI_Datasets/Aircraft/fgvc-aircraft-2013b/data/images_variant_trainval.txt'
        test_list = '/home/datasets/HI_Datasets/Aircraft/fgvc-aircraft-2013b/data/images_variant_test.txt'
        trees = [
            [100, 0, 30],
            [101, 0, 31],
            [102, 0, 32],
            [103, 0, 32],
            [104, 0, 32],
            [105, 0, 32],
            [106, 0, 33],
            [107, 0, 33],
            [108, 0, 34],
            [109, 0, 34],
            [110, 0, 34],
            [111, 0, 34],
            [112, 0, 35],
            [113, 1, 36],
            [114, 2, 37],
            [115, 2, 38],
            [116, 6, 39],
            [117, 6, 39],
            [118, 6, 40],
            [119, 3, 41],
            [120, 4, 42],
            [121, 4, 43],
            [122, 4, 44],
            [123, 4, 45],
            [124, 4, 45],
            [125, 4, 45],
            [126, 4, 45],
            [127, 4, 45],
            [128, 4, 45],
            [129, 4, 45],
            [130, 4, 45],
            [131, 4, 46],
            [132, 4, 46],
            [133, 4, 46],
            [134, 4, 46],
            [135, 4, 47],
            [136, 4, 47],
            [137, 4, 48],
            [138, 4, 48],
            [139, 4, 48],
            [140, 4, 49],
            [141, 4, 49],
            [142, 20, 50],
            [143, 13, 51],
            [144, 8, 52],
            [145, 8, 53],
            [146, 8, 54],
            [147, 8, 54],
            [148, 7, 55],
            [149, 7, 56],
            [150, 7, 57],
            [151, 7, 57],
            [152, 11, 58],
            [153, 11, 58],
            [154, 22, 59],
            [155, 13, 60],
            [156, 13, 61],
            [157, 13, 62],
            [158, 22, 63],
            [159, 11, 64],
            [160, 11, 65],
            [161, 11, 66],
            [162, 12, 67],
            [163, 25, 68],
            [164, 14, 69],
            [165, 14, 70],
            [166, 14, 70],
            [167, 14, 70],
            [168, 14, 71],
            [169, 14, 71],
            [170, 14, 72],
            [171, 15, 73],
            [172, 22, 74],
            [173, 21, 75],
            [174, 10, 76],
            [175, 10, 77],
            [176, 17, 78],
            [177, 17, 79],
            [178, 17, 80],
            [179, 5, 81],
            [180, 18, 82],
            [181, 18, 82],
            [182, 6, 83],
            [183, 19, 84],
            [184, 3, 85],
            [185, 20, 86],
            [186, 22, 87],
            [187, 22, 88],
            [188, 22, 88],
            [189, 22, 89],
            [190, 16, 90],
            [191, 24, 91],
            [192, 26, 92],
            [193, 26, 93],
            [194, 27, 94],
            [195, 9, 95],
            [196, 23, 96],
            [197, 28, 97],
            [198, 28, 98],
            [199, 29, 99]
        ]
        levels = 3
        total_nodes = 200
        trainset = AirDataset(data_dir, train_list, transform_train, re_level='family', proportion=args.proportion)
        testset = AirDataset(data_dir, test_list, transform_test, re_level='class', proportion=1.0)
    elif args.dataset == 'Car':
        # StandCars
        data_dir = '/home/datasets/HI_Datasets/StandCars/cars196'
        train_list = '/home/datasets/HI_Datasets/StandCars/car_train.txt'
        test_list = '/home/datasets/HI_Datasets/StandCars/car_test.txt'
        trees = [
            [9, 6],
            [10, 5],
            [11, 5],
            [12, 5],
            [13, 5],
            [14, 2],
            [15, 3],
            [16, 1],
            [17, 2],
            [18, 1],
            [19, 2],
            [20, 1],
            [21, 2],
            [22, 2],
            [23, 2],
            [24, 5],
            [25, 5],
            [26, 8],
            [27, 3],
            [28, 5],
            [29, 1],
            [30, 2],
            [31, 5],
            [32, 5],
            [33, 2],
            [34, 5],
            [35, 1],
            [36, 2],
            [37, 5],
            [38, 8],
            [39, 1],
            [40, 6],
            [41, 6],
            [42, 2],
            [43, 5],
            [44, 1],
            [45, 6],
            [46, 1],
            [47, 1],
            [48, 5],
            [49, 5],
            [50, 2],
            [51, 2],
            [52, 5],
            [53, 1],
            [54, 2],
            [55, 5],
            [56, 6],
            [57, 5],
            [58, 6],
            [59, 5],
            [60, 6],
            [61, 0],
            [62, 0],
            [63, 1],
            [64, 2],
            [65, 2],
            [66, 6],
            [67, 1],
            [68, 4],
            [69, 5],
            [70, 6],
            [71, 5],
            [72, 7],
            [73, 0],
            [74, 2],
            [75, 5],
            [76, 6],
            [77, 0],
            [78, 0],
            [79, 7],
            [80, 2],
            [81, 5],
            [82, 0],
            [83, 0],
            [84, 6],
            [85, 1],
            [86, 4],
            [87, 5],
            [88, 1],
            [89, 1],
            [90, 8],
            [91, 8],
            [92, 8],
            [93, 4],
            [94, 0],
            [95, 0],
            [96, 7],
            [97, 6],
            [98, 0],
            [99, 0],
            [100, 8],
            [101, 2],
            [102, 6],
            [103, 6],
            [104, 5],
            [105, 5],
            [106, 3],
            [107, 2],
            [108, 1],
            [109, 2],
            [110, 1],
            [111, 1],
            [112, 2],
            [113, 5],
            [114, 0],
            [115, 1],
            [116, 4],
            [117, 6],
            [118, 6],
            [119, 0],
            [120, 2],
            [121, 0],
            [122, 0],
            [123, 5],
            [124, 8],
            [125, 5],
            [126, 6],
            [127, 7],
            [128, 6],
            [129, 6],
            [130, 0],
            [131, 1],
            [132, 0],
            [133, 0],
            [134, 4],
            [135, 4],
            [136, 2],
            [137, 5],
            [138, 3],
            [139, 6],
            [140, 6],
            [141, 6],
            [142, 5],
            [143, 5],
            [144, 5],
            [145, 5],
            [146, 5],
            [147, 3],
            [148, 5],
            [149, 2],
            [150, 6],
            [151, 6],
            [152, 2],
            [153, 6],
            [154, 6],
            [155, 6],
            [156, 6],
            [157, 6],
            [158, 2],
            [159, 2],
            [160, 2],
            [161, 2],
            [162, 6],
            [163, 6],
            [164, 5],
            [165, 1],
            [166, 1],
            [167, 6],
            [168, 2],
            [169, 1],
            [170, 5],
            [171, 2],
            [172, 5],
            [173, 5],
            [174, 7],
            [175, 5],
            [176, 3],
            [177, 7],
            [178, 3],
            [179, 2],
            [180, 2],
            [181, 5],
            [182, 4],
            [183, 1],
            [184, 5],
            [185, 5],
            [186, 3],
            [187, 1],
            [188, 2],
            [189, 5],
            [190, 5],
            [191, 3],
            [192, 5],
            [193, 5],
            [194, 6],
            [195, 5],
            [196, 5],
            [197, 6],
            [198, 3],
            [199, 3],
            [200, 3],
            [201, 3],
            [202, 5],
            [203, 6],
            [204, 1]
        ]
        levels = 2
        total_nodes = 205
        trainset = CarDataset(data_dir, train_list, transform_train, re_level='family', proportion=1.0)
        testset = CarDataset(data_dir, test_list, transform_test, re_level='class', proportion=1.0)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    # GPU
    device = torch.device("cuda:" + args.device[0])
    
    # HMGN
    backbone = models.resnet50(pretrained=False)
    # backbone.load_state_dict(torch.load('./pre-trained/resnet50-19c8e357.pth'))
    # backbone = models.densenet161(pretrained=True)
    adjacency_matrix = create_adjacency_matrix(trees, total_nodes, levels)
    net = HMGN(backbone, image_feature_dim=2048, adjacency_matrix = adjacency_matrix, 
               word_features = './pre-trained/word_embeddings_CUB.pkl', device = device, num_classes = total_nodes)
    
    checkpoint = torch.load('./models_CUB/model_CUB_200_448_p1.0_bz16_HMGN_Cos_6L.pth.tar')
    net.load_state_dict(checkpoint['state_dict'])

    net.to(device)

    # Loss functions
    CELoss = nn.CrossEntropyLoss()
    tree = TreeLoss(trees, total_nodes, levels, device)
    
    if args.proportion > 0.1:       # for p > 0.1
        optimizer = optim.SGD([
            {'params': net.word_semantic.parameters(), 'lr': 0.002},
            {'params': net.gnn_cells.parameters(), 'lr': 0.002},
            {'params': net.fc.parameters(), 'lr': 0.002},
            {'params': net.backbone.parameters(), 'lr': 0.0002}
        ],
            momentum=0.9, weight_decay=5e-4)
    
    else:     # for p = 0.1
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
    
    save_name = args.dataset+'_'+str(args.epoch)+'_'+str(args.img_size)+'_p'+str(args.proportion)+'_bz'+str(args.batch)+'_HMGN'+'_'+args.lr_adjt+'_6L'
    train(nb_epoch, net, trainloader, testloader, optimizer, scheduler, args.lr_adjt, args.dataset, CELoss, tree, device, args.device, save_name)
    test(net, testloader, CELoss, tree, device, args.dataset)
