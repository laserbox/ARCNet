import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels
from efficientnet_pytorch import EfficientNet

from lib.models.MobileNetV2 import mobilenet_v2


class RA(nn.Module):
    def __init__(self, cnn_model_name='resnet18', input_size=224, hidden_size=256, layer_num=3, recurrent_num=5, class_num=5, dropout_p=0.5, pretrain=False):
        super(RA, self).__init__()

        self.cnn_model_name = cnn_model_name
        self.cnn, feature_dim, reduction = get_pretrained_cnn(
            cnn_model_name, class_num, pretrain)

        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.recurrent_num = recurrent_num
        self.mask_size = int(input_size / reduction)**2
        self.class_num = class_num

        self.lstm = nn.LSTM(self.feature_dim, self.hidden_size,
                            self.layer_num, batch_first=True)
        self.att = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.hidden_size, self.mask_size, bias=False),
            )
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.hidden_size, self.class_num, bias=False),
            )

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def single_lstm(self, input):
        # pdb.set_trace()
        input = (self.m.unsqueeze(1) * input).sum(2)
        input = input.unsqueeze(1)

        _, (self.h, self.c) = self.lstm(input, (self.h, self.c))

        output = self.tanh(self.h[-1])

        self.m = self.att(self.h[-1])
        # self.m = F.softmax(self.m, dim=1)
        self.m = self.sigmoid(self.m)

        output = self.fc(output)

        # output = F.softmax(output, dim=1)

        return output

    def forward(self, input):
        if 'efficient' in self.cnn_model_name:
            cnn_x = self.cnn.extract_features(input)  # 1,c,7,7
        elif 'vgg' in self.cnn_model_name:
            cnn_x = self.cnn._features(input)  # 1,512,7,7
        else:
            cnn_x = self.cnn.features(input)  # 1,c,7,7

        self.h = Variable(torch.zeros(
            self.layer_num, cnn_x.size(0), self.hidden_size)).cuda()
        self.c = Variable(torch.zeros(
            self.layer_num, cnn_x.size(0), self.hidden_size)).cuda()
        self.m = Variable(torch.ones(cnn_x.size(0), self.mask_size)).cuda()
        # self.m = F.softmax(self.m, dim=1)

        cnn_x = cnn_x.view(cnn_x.size(0), cnn_x.size(1), -1)

        o = 0
        for _ in range(self.recurrent_num):
            o += self.single_lstm(cnn_x)
        return o

    def get_config_optim(self, lr_cnn, lr_lstm):
        return [{'params': self.cnn.parameters(), 'lr': lr_cnn},
                {'params': self.att.parameters(), 'lr': lr_lstm},
                {'params': self.fc.parameters(), 'lr': lr_lstm},
                {'params': self.lstm.parameters(), 'lr': lr_lstm},
                ]


def get_pretrained_cnn(model_name='resnet18', output_num=None, pretrained=True, freeze_bn=False, dropout_p=0, **kwargs):

    if 'efficientnet' in model_name:
        model = EfficientNet.from_pretrained(
            model_name, num_classes=output_num)
        channel = model._fc.in_features

    elif 'densenet' in model_name:
        model = models.__dict__[model_name](
            num_classes=1000, pretrained=pretrained)
        channel = model.classifier.in_features
        model.classifier = nn.Linear(channel, output_num)

    elif 'mobilenet' in model_name:
        model = mobilenet_v2(pretrained=pretrained)
        channel = model.classifier.in_features
        model.classifier = nn.Linear(channel, output_num)

    else:
        pretrained = 'imagenet' if pretrained else None
        model = pretrainedmodels.__dict__[model_name](
            num_classes=1000, pretrained=pretrained)

        if 'dpn' in model_name:
            channel = model.last_linear.in_channels
            model.last_linear = nn.Conv2d(
                channel, output_num, kernel_size=1, bias=True)
        else:
            if 'resnet' in model_name:
                model.avgpool = nn.AdaptiveAvgPool2d(1)
            else:
                model.avg_pool = nn.AdaptiveAvgPool2d(1)
            channel = model.last_linear.in_features
            if dropout_p == 0:
                model.last_linear = nn.Linear(channel, output_num)
            else:
                model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout_p),
                    nn.Linear(channel, output_num),
                )
    if 'resnet' in model_name or 'se_resnext' in model_name:
        reduction = 32
    elif 'vgg' in model_name:
        reduction = 32
        channel = 512
    else:
        reduction = 32
    return model, channel, reduction
