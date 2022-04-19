import pickle
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
import scipy.sparse as sp
import math



class Element_Wise_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Element_Wise_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        for i in range(self.in_features):
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            for i in range(self.in_features):
                self.bias[i].data.uniform_(-stdv, stdv)
    
    def forward(self, input):
        x = input * self.weight
        x = torch.sum(x,2)
        if self.bias is not None:
            x = x + self.bias
        return x



class semantic(nn.Module):
    def __init__(self, num_classes, image_feature_dim, word_feature_dim, intermediary_dim=1024):
        super(semantic, self).__init__()
        self.num_classes = num_classes
        self.image_feature_dim = image_feature_dim
        self.word_feature_dim = word_feature_dim
        self.intermediary_dim = intermediary_dim
        self.fc_1 = nn.Linear(self.image_feature_dim, self.intermediary_dim, bias=False)
        self.fc_2 = nn.Linear(self.word_feature_dim, self.intermediary_dim, bias=False)
        self.fc_3 = nn.Linear(self.intermediary_dim, self.intermediary_dim)
        self.fc_a = nn.Linear(self.intermediary_dim, 1)

    def forward(self, batch_size, img_feature_map, word_features):
        convsize = img_feature_map.size()[3]

        img_feature_map = torch.transpose(torch.transpose(img_feature_map, 1, 2),2,3)
        f_wh_feature = img_feature_map.contiguous().view(batch_size*convsize*convsize, -1)
        f_wh_feature = self.fc_1(f_wh_feature).view(batch_size*convsize*convsize, 1, -1).repeat(1, self.num_classes, 1)

        f_wd_feature = self.fc_2(word_features).view(1, self.num_classes, 1024).repeat(batch_size*convsize*convsize,1,1)
        lb_feature = self.fc_3(torch.tanh(f_wh_feature*f_wd_feature).view(-1,1024))
        coefficient = self.fc_a(lb_feature)
        coefficient = torch.transpose(torch.transpose(coefficient.view(batch_size, convsize, convsize, self.num_classes),2,3),1,2).view(batch_size, self.num_classes, -1)

        coefficient = F.softmax(coefficient, dim=2)
        coefficient = coefficient.view(batch_size, self.num_classes, convsize, convsize)
        coefficient = torch.transpose(torch.transpose(coefficient,1,2),2,3)
        coefficient = coefficient.view(batch_size, convsize, convsize, self.num_classes, 1).repeat(1,1,1,1,self.image_feature_dim)
        img_feature_map = img_feature_map.view(batch_size, convsize, convsize, 1, self.image_feature_dim).repeat(1, 1, 1, self.num_classes, 1) * coefficient
        graph_net_input = torch.sum(torch.sum(img_feature_map,1) ,1)
        return graph_net_input


class GatedGNN(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(GatedGNN, self).__init__()

        # conv1
        self.Ui1 = nn.Linear(dim_in, dim_out, bias=False)
        self.Uj1 = nn.Linear(dim_in, dim_out, bias=False)
        self.Vi1 = nn.Linear(dim_in, dim_out, bias=False)
        self.Vj1 = nn.Linear(dim_in, dim_out, bias=False)
        self.bu1 = torch.nn.Parameter( torch.FloatTensor(dim_out), requires_grad=True )
        self.bv1 = torch.nn.Parameter( torch.FloatTensor(dim_out), requires_grad=True )

        # conv2
        self.Ui2 = nn.Linear(dim_out, dim_out, bias=False)
        self.Uj2 = nn.Linear(dim_out, dim_out, bias=False)
        self.Vi2 = nn.Linear(dim_out, dim_out, bias=False)
        self.Vj2 = nn.Linear(dim_out, dim_out, bias=False)
        self.bu2 = torch.nn.Parameter( torch.FloatTensor(dim_out), requires_grad=True )
        self.bv2 = torch.nn.Parameter( torch.FloatTensor(dim_out), requires_grad=True )

        # bn1, bn2
        self.bn1 = torch.nn.BatchNorm1d(dim_out)
        self.bn2 = torch.nn.BatchNorm1d(dim_out)

        # resnet
        self.R = nn.Linear(dim_in, dim_out, bias=False)

        # init
        self.init_weights_OurConvNetcell(dim_in, dim_out, 1)


    def init_weights_OurConvNetcell(self, dim_in, dim_out, gain):

        # conv1
        scale = gain* np.sqrt( 2.0/ dim_in )
        self.Ui1.weight.data.uniform_(-scale, scale)
        self.Uj1.weight.data.uniform_(-scale, scale)
        self.Vi1.weight.data.uniform_(-scale, scale)
        self.Vj1.weight.data.uniform_(-scale, scale)
        scale = gain* np.sqrt( 2.0/ dim_out )
        self.bu1.data.fill_(0)
        self.bv1.data.fill_(0)

        # conv2
        scale = gain* np.sqrt( 2.0/ dim_out )
        self.Ui2.weight.data.uniform_(-scale, scale)
        self.Uj2.weight.data.uniform_(-scale, scale)
        self.Vi2.weight.data.uniform_(-scale, scale)
        self.Vj2.weight.data.uniform_(-scale, scale)
        scale = gain* np.sqrt( 2.0/ dim_out )
        self.bu2.data.fill_(0)
        self.bv2.data.fill_(0)

        # RN
        scale = gain* np.sqrt( 2.0/ dim_in )
        self.R.weight.data.uniform_(-scale, scale)


    def forward(self, x, E_start, E_end):

        # E_start, E_end : E x V
        batch_size = x.size()[0]
        num_nodes = x.size()[1]
        batch_E_end = E_end.repeat(batch_size, 1).view(batch_size, -1 , num_nodes)
        batch_E_start = E_start.repeat(batch_size, 1).view(batch_size, -1, num_nodes)

        xin = x

        # conv1
        Vix = self.Vi1(x)  #  V x H_out
        Vjx = self.Vj1(x)  #  V x H_out
        # x1 = torch.mm(E_end, Vix) + torch.mm(E_start, Vjx) + self.bv1  # E x H_out
        x1 = torch.bmm(batch_E_end, Vix) + torch.bmm(batch_E_start, Vjx) + self.bv1
        x1 = torch.sigmoid(x1)
        Ujx = self.Uj1(x)  #  V x H_out
        # x2 = torch.mm(E_start, Ujx)  #  V x H_out
        x2 = torch.bmm(batch_E_start, Ujx)
        Uix = self.Ui1(x)  #  V x H_out
        # x = Uix + torch.mm(E_end.t(), x1*x2) + self.bu1 #  V x H_out
        x = Uix + torch.bmm(torch.transpose(batch_E_end, 2, 1), x1*x2) + self.bu1
        # bn1
        x = torch.transpose(self.bn1(torch.transpose(x,1,2)),1,2)
        # relu1
        x = F.relu(x)

        # conv2
        Vix = self.Vi2(x)  #  V x H_out
        Vjx = self.Vj2(x)  #  V x H_out
        # x1 = torch.mm(E_end,Vix) + torch.mm(E_start,Vjx) + self.bv2  # E x H_out
        x1 = torch.bmm(batch_E_end, Vix) + torch.bmm(batch_E_start, Vjx) + self.bv2
        x1 = torch.sigmoid(x1)
        Ujx = self.Uj2(x)  #  V x H_out
        # x2 = torch.mm(E_start, Ujx)  #  V x H_out
        x2 = torch.bmm(batch_E_start, Ujx)
        Uix = self.Ui2(x)  #  V x H_out
        # x = Uix + torch.mm(E_end.t(), x1*x2) + self.bu2 #  V x H_out
        x = Uix + torch.bmm(torch.transpose(batch_E_end, 2, 1), x1*x2) + self.bu2
        # bn2
        x = torch.transpose(self.bn2(torch.transpose(x,1,2)),1,2)

        # addition
        x = x + self.R(xin)
        # relu2
        x = F.relu(x)

        return x



class HMGN(nn.Module):
    def __init__(self, backbone, image_feature_dim, adjacency_matrix, word_features, device, num_classes = 80, word_feature_dim = 300):
        super(HMGN, self).__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim

        self.word_semantic = semantic(num_classes = self.num_classes, 
                                      image_feature_dim = self.image_feature_dim,
                                      word_feature_dim = self.word_feature_dim)
        self._word_features = self.load_features(word_features)
        self.E_start, self.E_end = self.generate_E_start_end(adjacency_matrix)

        list_of_gnn_cells = []
        for layer in range(3):
            list_of_gnn_cells.append(GatedGNN(image_feature_dim, image_feature_dim))
        self.gnn_cells = nn.ModuleList(list_of_gnn_cells)
        self.fc = Element_Wise_Layer(self.num_classes, self.image_feature_dim)


    def forward(self, x):
        batch_size = x.size()[0]
        feature_map = self.backbone(x)
        # feature_fc = self.pooling(feature_map)
        # feature_fc = feature_fc.view(feature_fc.size(0), -1)
        graph_net_input = self.word_semantic(batch_size, feature_map, self._word_features.cuda())

        for layer in range(3):
            gnn_layer = self.gnn_cells[layer]
            graph_net_input = gnn_layer(graph_net_input, self.E_start.cuda(), self.E_end.cuda())
        # FC layers
        x = self.fc(graph_net_input)
        y_order_sig = torch.sigmoid(x[:, :13])
        y_family_sig = torch.sigmoid(x[:, 13:51])
        # y_species_sof = torch.softmax(x[:, 51:], dim=1)
        y_species_sof = x[:, 51:]
        y_species_sig = torch.sigmoid(x[:, 51:])
        return y_order_sig, y_family_sig, y_species_sof, y_species_sig


    def load_features(self, word_features):
        return torch.from_numpy(pickle.load(open(word_features, 'rb')).astype(np.float32))


    def generate_E_start_end(self, adjacency_matrix):
        A = adjacency_matrix
        # Self node loop
        for i in range(A.shape[1]):
            A[i%A.shape[0],i]=i//A.shape[0]+1
        W_coo = sp.coo_matrix(A)
        nb_edges = W_coo.nnz
        nb_vertices = A.shape[0]
        edge_to_starting_vertex = sp.coo_matrix((W_coo.data, (np.arange(nb_edges), W_coo.row)), shape=(nb_edges, nb_vertices))
        new_col = np.where(W_coo.col >= nb_vertices, W_coo.col % nb_vertices, W_coo.col)
        edge_to_ending_vertex = sp.coo_matrix((W_coo.data, (np.arange(nb_edges), new_col)), shape=(nb_edges, nb_vertices))
        E_start = edge_to_starting_vertex
        E_end   = edge_to_ending_vertex
        E_start = torch.from_numpy(E_start.toarray()).to(torch.float32)
        E_end = torch.from_numpy(E_end.toarray()).to(torch.float32)

        return E_start, E_end