from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils import model_zoo

import torch.nn.functional as F
from torchvision import models
import pretrainedmodels
from efficientnet_pytorch import EfficientNet

from lib.models.MobileNetV2 import mobilenet_v2

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SoftLabelGCN(nn.Module):
    def __init__(self, cnn_model_name='resnet18', cnn_pretrained=True, num_outputs=5, **kwargs):
        super(SoftLabelGCN, self).__init__()
        self.cnn_model_name = cnn_model_name
        self.cnn, feature_dim = get_cnn_model(cnn_model_name, num_outputs, cnn_pretrained)

        hidden_dim = int(feature_dim/2)

        self.gcn1 = GraphConvolution(num_outputs, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, feature_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

        self.register_parameter('adj', nn.Parameter(torch.tensor(get_gcn_adj(num_outputs), dtype=torch.float)))
        self.register_buffer('adj_mask', torch.tensor(get_adj_mask(num_outputs), dtype=torch.float))
        self.register_buffer('inp', torch.tensor(get_gcn_inp(num_outputs), dtype=torch.float))
        self.register_buffer('diag', torch.eye(num_outputs, dtype=torch.float))

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.cnn_linear = nn.Linear(feature_dim, num_outputs)

        self.remove_gcngate = False

    def forward(self, input):
        temp = self.adj * self.adj_mask.detach()
        temp = temp + temp.t()
        a = self.relu(temp) + self.diag.detach()

        D = torch.pow(a.sum(1).float(), -0.5)
        D = torch.diag(D)
        A = torch.matmul(torch.matmul(a, D).t(), D)

        x = self.gcn1(self.inp.detach(), A)
        x = self.leaky_relu(x)
        x = self.gcn2(x, A.detach())
        x = x.transpose(0, 1)
        # x = self.dropout(x)

        if 'efficient' in self.cnn_model_name:
            cnn_x = self.cnn.extract_features(input)

        else:
            cnn_x = self.cnn.features(input)
        
        if 'vgg' in self.cnn_model_name:
            cnn_x = self.cnn.relu1(cnn_x)
            cnn_x = self.cnn.dropout1(cnn_x)
            cnn_x = cnn_x.view(cnn_x.size(0), -1)
        else:
            cnn_x = self.avg(cnn_x)
            cnn_x = cnn_x.view(cnn_x.size(0), -1)

        # cnn_x = self.dropout(cnn_x)

        x = torch.matmul(cnn_x.detach(), x)
        
        cnn_x = self.cnn_linear(cnn_x)

        x = self.sigmoid(x)

        if self.remove_gcngate:
            x = 0

        return x * cnn_x + cnn_x, temp
    
    def get_config_optim(self, lr_cnn, lr_gcn, lr_adj):
        return [{'params': self.cnn.parameters(), 'lr': lr_cnn},
                {'params': self.cnn_linear.parameters(), 'lr': lr_cnn},
                {'params': self.gcn1.parameters(), 'lr': lr_gcn},
                {'params': self.gcn2.parameters(), 'lr': lr_gcn},
                {'params': self.adj, 'lr': lr_adj},
                ]

def get_cnn_model(model_name='resnet18', num_outputs=None, pretrained=True,
              freeze_bn=False, dropout_p=0, **kwargs):

    if 'efficientnet' in model_name:
        model = EfficientNet.from_pretrained(model_name, num_classes=num_outputs)
        in_features = model._fc.in_features

    elif 'densenet' in model_name:
        model = models.__dict__[model_name](num_classes=1000,
                                            pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_outputs)

    elif 'mobilenet' in model_name:
        model = mobilenet_v2(pretrained=pretrained)
        in_features = model.classifier.in_features        

    else:
        pretrained = 'imagenet' if pretrained else None
        model = pretrainedmodels.__dict__[model_name](num_classes=1000,
                                                      pretrained=pretrained)

        if 'dpn' in model_name:
            in_features = model.last_linear.in_channels
            model.last_linear = nn.Conv2d(in_features, num_outputs,
                                          kernel_size=1, bias=True)
        else:
            if 'resnet' in model_name:
                model.avgpool = nn.AdaptiveAvgPool2d(1)
            else:
                model.avg_pool = nn.AdaptiveAvgPool2d(1)
            in_features = model.last_linear.in_features
            if dropout_p == 0:
                model.last_linear = nn.Linear(in_features, num_outputs)
            else:
                model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout_p),
                    nn.Linear(in_features, num_outputs),
                )

    return model, in_features

def get_gcn_inp(num):
    inp = np.eye(num)
    return inp

def get_gcn_adj(num):
    adj = (np.ones([num,num]) - np.eye(num))
    # adj = np.zeros([num,num])
    return adj

def get_adj_mask(num):
    mask=0
    for i in range(num-1):
        mask += np.eye(num, num, -i-1)
    return mask
