from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import model
import pdb
import vgg
import resnet
import time
if __name__ == '__main__':
	CNN='vgg'
	DATASET='my'	
	num_epochs=1

	if DATASET == 'ucm-80%':
		num_classes=21
		batch_size=1
	elif DATASET == 'ucm-50%':
		num_classes=21
		batch_size=10
	elif DATASET == 'rs-60%':
		num_classes=19
		batch_size=95
	elif DATASET == 'rs-40%':
		num_classes=19
		batch_size=10
	elif DATASET == 'my':
		num_classes=31
		batch_size=12
	elif DATASET == 'nwpu':
		num_classes=45
		batch_size=30
	elif DATASET == 'aid-50%':
		num_classes=30
		batch_size=50
	elif DATASET == 'aid-20%':
		num_classes=30
		batch_size=50
	if CNN == 'alex':
		LR=0.0001
		input_size=256
		mask_size=36
		hidden_size=256
		num_layers=3
		num_recurrence=20
		cnn_model = model.AlexNet()
		cnn_model.load_state_dict(torch.load('alexnet-2.pth'))
		for param in cnn_model.parameters():
				param.requires_grad = False
	elif CNN == 'vgg':
		LR=0.0001
		input_size=512
		mask_size=49
		hidden_size=256
		num_layers=3
		num_recurrence=20		
		cnn_model = vgg.vgg16(pretrained=False)
		cnn_model.load_state_dict(torch.load('vgg-my-80%.pth'))
		for param in cnn_model.parameters():
				param.requires_grad = False
	elif CNN == 'google':
		LR=0.0001
		input_size=256
		mask_size=36
		hidden_size=256
		num_layers=3
		num_recurrence=30	
		cnn_model = model.AlexNet()
		cnn_model.load_state_dict(torch.load('alexnet-2.pth'))
		for param in cnn_model.parameters():
				param.requires_grad = False
	elif CNN == 'res':
		LR=0.0001
		input_size=512
		mask_size=49
		hidden_size=256
		num_layers=3
		num_recurrence=20	
		cnn_model = resnet.resnet34(pretrained=False)
		cnn_model.load_state_dict(torch.load('resnet.pth'))		
		for param in cnn_model.parameters():
				param.requires_grad = False	


	lstm_model = model.ALSTM(input_size, mask_size, hidden_size, num_layers, num_recurrence, num_classes, batch_size)
	lstm_model = (torch.load('/home/aaron/Desktop/my_attention/save_model/vgg/model-my.pth'))
	for param in lstm_model.parameters():
			param.requires_grad = False	
	lstm_model = model.test_model(cnn_model, lstm_model, num_recurrence, num_epochs, batch_size, DATASET)
