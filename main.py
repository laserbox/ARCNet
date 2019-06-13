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

if __name__ == '__main__':
	CNN='vgg'
	DATASET='aid-20%'	
	num_epochs=120

	if DATASET == 'ucm-80%':
		num_classes=21
		batch_size=60
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
		cnn_model.load_state_dict(torch.load('vgg-aid-20%.pth'))
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


	if CNN == 'alex':
		save_model_path='save_model/alexnet/model-temp.pth'
		f = open('save_model/alexnet/log.txt', 'w')		
	elif CNN == 'vgg':
		save_model_path='save_model/vgg/model-aid-20.pth'
		f = open('save_model/vgg/log.txt', 'w')			
	elif CNN == 'google':
		save_model_path='save_model/google/model-temp.pth'
		f = open('save_model/google/log.txt', 'w')			
	elif CNN == 'res':
		save_model_path='save_model/resnet/model-temp.pth'
		f = open('save_model/resnet/log.txt', 'w')


	lstm_model = model.ALSTM(input_size, mask_size, hidden_size, num_layers, num_recurrence, num_classes, batch_size)
	# lstm_model.load_state_dict(torch.load('/home/aaron/Desktop/my_attention/save_model/vgg/temp.pth'))
	lstm_params = list(filter(lambda p:p.requires_grad,lstm_model.parameters()))
	optimizer_ft = optim.Adam(lstm_params, weight_decay=5e-4, lr=LR)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)
	criterion = nn.CrossEntropyLoss()

	lstm_model = model.train_model(cnn_model, lstm_model, criterion, optimizer_ft, exp_lr_scheduler, num_recurrence, num_epochs, batch_size, DATASET, save_model_path)

	print('hidden_size: {:4f}'.format(hidden_size))
	print('num_recurrence: {:4f}'.format(num_recurrence))
	print('LR: {:4f}'.format(LR))
	print('batch_size: {:4f}'.format(batch_size))
	print(DATASET)
	print(CNN)

	# if CNN == 'alex':
	# 	torch.save(lstm_model, 'save_model/alexnet/model-temp.pth')
	# 	torch.save(lstm_model.state_dict(), 'save_model/alexnet/temp.pth')
	# 	f = open('save_model/alexnet/log.txt', 'w')		
	# elif CNN == 'vgg':
	# 	torch.save(lstm_model, 'save_model/vgg/model-temp1.pth')
	# 	torch.save(lstm_model.state_dict(), 'save_model/vgg/temp1.pth')
	# 	f = open('save_model/vgg/log.txt', 'w')			
	# elif CNN == 'google':
	# 	torch.save(lstm_model, 'save_model/google/model-temp.pth')
	# 	torch.save(lstm_model.state_dict(), 'save_model/google/temp.pth')
	# 	f = open('save_model/google/log.txt', 'w')			
	# elif CNN == 'res':
	# 	torch.save(lstm_model, 'save_model/resnet/model-temp.pth')
	# 	torch.save(lstm_model.state_dict(), 'save_model/resnet/temp.pth')
	# 	f = open('save_model/resnet/log.txt', 'w')
	f.write('num_layers: {:4f} \n'.format(num_layers))			
	f.write('hidden_size: {:4f} \n'.format(hidden_size))
	f.write('num_recurrence: {:4f} \n'.format(num_recurrence))
	f.write('LR: {:4f} \n'.format(LR))
	f.write('batch_size: {:4f} \n'.format(batch_size))
	f.write(DATASET)
	f.write('\n')
	f.write(CNN)
	f.close()						