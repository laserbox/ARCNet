import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional
from torch import optim
import numpy as np
from torchvision import datasets, models, transforms
import os
import time
import pdb
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
class AlexNet(nn.Module):

	def __init__(self, num_classes=1000):
		super(AlexNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		# x = x.view(x.size(0), 256 * 6 * 6)
		# x = self.classifier(x)
		return x

# class Attention(nn.Module):
# 	"""docstring for ClassName"""
# 	def __init__(self, input_size, mask_size, hidden_size, num_layers, batch_size):
# 		super(Attention, self).__init__()
# 		self.input_size = input_size
# 		self.hidden_size = hidden_size
# 		self.num_layers = num_layers
# 		self.batch_size = batch_size
# 		self.mask_size = mask_size
# 		self.atten = nn.Linear(self.hidden_size, self.mask_size, bias=False)
# 		self.mask = self.mask_init()
# 	def forward(self, hidden):
# 		#input = input.view(self.batch_size,self.input_size,-1)
# 		hidden = hidden[self.num_layers-1]
# 		self.mask = self.atten(hidden)
# 		self.mask = nn.functional.softmax(mask)
# 		#masked_output = (mask.unsqueeze(1)*input).sum(2)

# 		return self.mask
# 	def mask_init(self)
# 		m0 = Variable(torch.ones(self.batch_size,self.mask_size) / self.mask_size)
# 		return m0.cuda()

# class Lstm(nn.Module):
# 	"""docstring for ClassName"""
# 	def __init__(self, input_size, mask_size, hidden_size, num_layers, num_recurrence, num_classes, batch_size):
# 		super(Lstm, self).__init__()
# 		self.input_size = input_size
# 		self.hidden_size = hidden_size
# 		self.num_layers = num_layers
# 		self.num_recurrence = num_recurrence
# 		self.batch_size = batch_size
# 		self.mask_size = mask_size
# 		self.num_classes = num_classes
# 		self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
# 		self.atten_mask = Variable(torch.zeros(self.num_recurrence, self.batch_size, 36))
# 		self.atten = Attention(self.input_size, self.mask_size, self.hidden_size, self.num_layers, self.batch_size)
# 		self.fc = nn.Linear(self.hidden_size, self.num_classes, bias=False)
# 		self.hidden, self.cell = self.hidden_init()
# 		self.tanh = nn.Tanh()
# 	def forward(self, input, hidden, cell):
# 		#lstm_in, _ = self.atten(input, self.hidden)
# 		lstm_in = lstm_in.unsqueeze(1)
# 		output, (self.hidden, self.cell) = self.lstm(lstm_in, (self.hidden, self.cell))
# 		output = self.tanh(output)
# 		output = self.fc(output)
# 		output = nn.functional.softmax(output)
# 		return output, hidden, cell
# 	# def forward(self, input):
# 	# 	for i in range(self.num_recurrence):
# 	# 		lstm_in, self.atten_mask[i-1] = self.atten(input, self.hidden)
# 	# 		lstm_in = lstm_in.unsqueeze(1)
# 	# 		output, (self.hidden, self.cell) = self.lstm(lstm_in, (self.hidden, self.cell))
# 	# 	output = self.tanh(output)
# 	# 	output = self.fc(output)
# 	# 	output = nn.functional.softmax(output)
# 	# 	return output
# 	def hidden_init(self, use_cuda=True):
# 		h0 = Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size))
# 		c0 = Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size))
		
# 		if use_cuda:
# 			return h0.cuda(), c0.cuda()
# 		else:
# 			return h0, c0


class ALSTM(nn.Module):
	"""docstring for ClassName"""
	def __init__(self, input_size, mask_size, hidden_size, num_layers, num_recurrence, num_classes, batch_size):
		super(ALSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.num_recurrence = num_recurrence
		self.batch_size = batch_size
		self.mask_size = mask_size
		self.num_classes = num_classes
		self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
		self.atten = nn.Linear(self.hidden_size, self.mask_size, bias=False)
		self.fc = nn.Linear(self.hidden_size, self.num_classes, bias=False)
		self.hidden, self.cell = self.hidden_init()
		self.mask = self.mask_init()
		self.tanh = nn.Tanh()
		self.drop = nn.Dropout()
	def forward(self, input):
		input = input.view(self.batch_size,self.input_size,-1)
		input = (self.mask.unsqueeze(1)*input).sum(2)
		input = input.unsqueeze(1)
		output, (self.hidden, self.cell) = self.lstm(input, (self.hidden, self.cell))
		output = self.tanh(self.hidden[self.num_layers-1])
		#output = self.tanh(output)
		# output = self.drop(output)
		output = self.fc(output)
		output = nn.functional.softmax(output)
		self.mask = self.atten(self.hidden[self.num_layers-1])
		self.mask = nn.functional.softmax(self.mask)
		return output
	# def forward(self, input):
	# 	for i in range(self.num_recurrence):
	# 		lstm_in, self.atten_mask[i-1] = self.atten(input, self.hidden)
	# 		lstm_in = lstm_in.unsqueeze(1)
	# 		output, (self.hidden, self.cell) = self.lstm(lstm_in, (self.hidden, self.cell))
	# 	output = self.tanh(output)
	# 	output = self.fc(output)
	# 	output = nn.functional.softmax(output)
	# 	return output
	def hidden_init(self, use_cuda=True):
		h0 = Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size))
		c0 = Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size))
		
		if use_cuda:
			return h0.cuda(), c0.cuda()
		else:
			return h0, c0

	def mask_init(self):
		m0 = Variable(torch.ones(self.batch_size,self.mask_size) / self.mask_size)
		return m0.cuda()


def train_model(cnn_model, lstm_model, criterion, optimizer, 
				scheduler, num_recurrence, num_epochs, batch_size, DATASET ,save_model_path):
	data_transforms = {
	  'train': transforms.Compose([
		  transforms.Scale(256),
		  transforms.RandomSizedCrop(224),
		  transforms.RandomHorizontalFlip(),
		  transforms.ToTensor(),
		  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		  ]),
	  'val': transforms.Compose([
		  transforms.Scale(256),
		  transforms.CenterCrop(224),
		  transforms.ToTensor(),
		  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		  ]),
	}
	if DATASET == 'ucm-80%':
		data_dir ='/home/aaron/Desktop/scene_dataset/UCMerced_LandUse/80%'
		# data_dir = '/home/aaron/Desktop/my_attention/dataset'
		# data_dir = '/home/aaron/Desktop/scene_dataset/UCMerced_LandUse/uc_cross_val_1'
		num_classes=21
	elif DATASET == 'ucm-50%':
		data_dir = '/home/aaron/Desktop/scene_dataset/UCMerced_LandUse/50%'
		num_classes=21		
	elif DATASET == 'rs-60%':
		data_dir = '/home/aaron/Desktop/scene_dataset/RS19/60%'
		num_classes=19
	elif DATASET == 'rs-40%':
		data_dir = '/home/aaron/Desktop/scene_dataset/RS19/40%'
		num_classes=19		
	elif DATASET == 'my':
		data_dir = '/home/aaron/Desktop/scene_dataset/my_scene_dataset/80%'
		num_classes=31
	elif DATASET == 'nwpu':
		data_dir = '/home/aaron/Desktop/scene_dataset/NWPU-RESISC45/nwpu_cross_val_1'
		num_classes=45
	elif DATASET == 'aid-50%':
		data_dir = '/home/aaron/Desktop/scene_dataset/AID/50%'
		num_classes=30
	elif DATASET == 'aid-20%':
		data_dir = '/home/aaron/Desktop/scene_dataset/AID/20%'
		num_classes=30
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
											data_transforms[x])
				  for x in ['train', 'val']}
	dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
											 shuffle=True, num_workers=0, drop_last=False)
				  for x in ['train', 'val']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
	class_names = image_datasets['train'].classes

	use_gpu = torch.cuda.is_available()
	# f = open('log1.txt', 'w')
	if use_gpu:
		cnn_model = cnn_model.cuda()
		lstm_model = lstm_model.cuda()
		criterion = criterion.cuda()
	#h0, c0 = lstm_model.hidden_init()
	since = time.time()

	best_model_wts = lstm_model.state_dict()
	best_acc = 0.0
	#writer = SummaryWriter()

	t = np.arange(1,num_epochs,1)
	plot_loss = np.zeros(num_epochs-1)

	for epoch in range(num_epochs):
		print('Epoch {}/{} \n'.format(epoch, num_epochs - 1))
		print('-' * 10)
		# f.write('Epoch {}/{} \n'.format(epoch, num_epochs - 1))
		# f.write('-' * 10)
		# f.write('\n')
		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				scheduler.step()
				lstm_model.train(True)  # Set model to training mode
			else:
				lstm_model.train(False)  # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0
			total_dataset = 0
			# Iterate over data.
			for data in dataloders[phase]:
				# get the inputs
				inputs, labels = data
				lstm_model.zero_grad()
				lstm_model.hidden, lstm_model.cell = lstm_model.hidden_init()
				lstm_model.mask = lstm_model.mask_init()

				# wrap them in Variable
				if use_gpu:
					inputs = Variable(inputs.cuda())
					labels = Variable(labels.cuda())
				else:
					inputs, labels = Variable(inputs), Variable(labels)

				# zero the parameter gradients
				optimizer.zero_grad()
				outputs = Variable(torch.zeros(batch_size, num_classes).cuda())
				# forward
				lstm_in = cnn_model(inputs)
				# pdb.set_trace()
				for i in range(num_recurrence):
					result = lstm_model(lstm_in).squeeze()
					# result = result * (i / num_recurrence)
					# outputs = result
					# pdb.set_trace()
					outputs = torch.add(outputs, result)
				# pdb.set_trace()
				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)

				# backward + optimize only if in training phase
				if phase == 'train':
					loss.backward()
					#torch.nn.utils.clip_grad_norm(lstm_model.parameters(), 10)
					optimizer.step()

				# statistics
				running_loss += loss.data[0]
				running_corrects += torch.sum(preds == labels.data)
				# pdb.set_trace()
				total_dataset += batch_size
			# pdb.set_trace()
			epoch_loss = running_loss / total_dataset
			epoch_acc = running_corrects / total_dataset
			if phase == 'train':
				plot_loss[epoch-1] = epoch_loss
			#writer.add_scalar('logs/loss', epoch_loss, epoch)
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))
			# f.write('{} Loss: {:.4f} Acc: {:.4f} \n'.format(
			# 	phase, epoch_loss, epoch_acc))

			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_epoch = epoch
				best_acc = epoch_acc
				torch.save(lstm_model, save_model_path)
				# best_model_wts = lstm_model.state_dict()

		print()


	#writer.export_scalars_to_json("./all_scalars.json")

	#writer.close()
	# f.close()
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))
	print('Best epoch: {:4f}'.format(best_epoch))

	plt.figure(1)
	plt.title('LSTM Layers : 5')
	# plt.title('Combine Approach : {:4f}'.format(5))
	plt.plot(t, plot_loss)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()
	# load best model weights
	# lstm_model = best_model_wts
	# lstm_model.load_state_dict(best_model_wts)
	return lstm_model

def test_model(cnn_model, lstm_model, num_recurrence, num_epochs, batch_size, DATASET):
	data_transforms = {
	  'train': transforms.Compose([
		  transforms.Scale(256),
		  transforms.RandomSizedCrop(224),
		  transforms.RandomHorizontalFlip(),
		  transforms.ToTensor(),
		  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		  ]),
	  'val': transforms.Compose([
		  transforms.Scale(256),
		  transforms.CenterCrop(224),
		  transforms.ToTensor(),
		  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		  ]),
	}
	if DATASET == 'ucm-80%':
		data_dir ='/home/aaron/Desktop/scene_dataset/UCMerced_LandUse/80%'
		# data_dir = '/home/aaron/Desktop/my_attention/dataset'
		# data_dir = '/home/aaron/Desktop/scene_dataset/UCMerced_LandUse/uc_cross_val_1'
		num_classes=21
	elif DATASET == 'ucm-50%':
		data_dir = '/home/aaron/Desktop/scene_dataset/UCMerced_LandUse/50%'
		num_classes=21		
	elif DATASET == 'rs-60%':
		data_dir = '/home/aaron/Desktop/scene_dataset/RS19/60%'
		num_classes=19
	elif DATASET == 'rs-40%':
		data_dir = '/home/aaron/Desktop/scene_dataset/RS19/40%'
		num_classes=19		
	elif DATASET == 'my':
		data_dir = '/home/aaron/Desktop/scene_dataset/my_scene_dataset/80%'
		num_classes=31
	elif DATASET == 'nwpu':
		data_dir = '/home/aaron/Desktop/scene_dataset/NWPU-RESISC45/nwpu_cross_val_1'
		num_classes=45
	elif DATASET == 'aid-50%':
		data_dir = '/home/aaron/Desktop/scene_dataset/AID/50%'
		num_classes=30
	elif DATASET == 'aid-20%':
		data_dir = '/home/aaron/Desktop/scene_dataset/AID/20%'
		num_classes=30
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
											data_transforms[x])
				  for x in ['train', 'val']}
	dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
											 shuffle=True, num_workers=0, drop_last=False)
				  for x in ['train', 'val']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
	class_names = image_datasets['train'].classes

	use_gpu = torch.cuda.is_available()
	if use_gpu:
		cnn_model = cnn_model.cuda()
		lstm_model = lstm_model.cuda()
	since = time.time()
	cm_preds=[1]
	cm_trues=[1]
	for epoch in range(num_epochs):
		print('Epoch {}/{} \n'.format(epoch, num_epochs - 1))
		print('-' * 10)
		# f.write('Epoch {}/{} \n'.format(epoch, num_epochs - 1))
		# f.write('-' * 10)
		# f.write('\n')
		# Each epoch has a training and validation phase
		lstm_model.train(False)  # Set model to evaluate mode

		running_corrects = 0
		total_dataset = 0
		# Iterate over data.
		for data in dataloders['val']:
			# get the inputs
			inputs, labels = data
			lstm_model.hidden, lstm_model.cell = lstm_model.hidden_init()
			lstm_model.mask = lstm_model.mask_init()
			# wrap them in Variable
			if use_gpu:
				inputs = Variable(inputs.cuda())
				labels = Variable(labels.cuda())
			else:
				inputs, labels = Variable(inputs), Variable(labels)
			outputs = Variable(torch.zeros(batch_size, num_classes).cuda())
			# forward
			lstm_in = cnn_model(inputs)
			# pdb.set_trace()
			for i in range(num_recurrence):
				result = lstm_model(lstm_in).squeeze()
				# result = result * (i / num_recurrence)
				# outputs = result
				# pdb.set_trace()
				outputs = torch.add(outputs, result)
			# pdb.set_trace()
			_, preds = torch.max(outputs.data, 1)
			# statistics
			running_corrects += torch.sum(preds == labels.data)
			# pdb.set_trace()
			total_dataset += batch_size
			cm_preds = np.r_[cm_preds, preds.cpu().numpy()]
			cm_trues = np.r_[cm_trues, labels.data.cpu().numpy()]
		# pdb.set_trace()
		# plot_confusion_matrix(cm_trues, cm_preds, class_names)
		epoch_acc = running_corrects / total_dataset
		print('val Acc: {:.4f}'.format(epoch_acc))
		print()
	time_elapsed = time.time() - since
	print(time_elapsed)
	return lstm_model


def plot_confusion_matrix(y_true, y_pred, labels):
	from sklearn.metrics import confusion_matrix
	cmap = plt.cm.binary
	cm = confusion_matrix(y_true, y_pred)
	tick_marks = np.array(range(len(labels))) + 0.5
	np.set_printoptions(precision=2)
	cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.figure(figsize=(10, 8), dpi=120)
	ind_array = np.arange(len(labels))
	x, y = np.meshgrid(ind_array, ind_array)
	intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
	for x_val, y_val in zip(x.flatten(), y.flatten()):
		#

		if (intFlag):
			c = cm[y_val][x_val]
			plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')

		else:
			c = cm_normalized[y_val][x_val]
			if (c > 0.01):
				#这里是绘制数字，可以对数字大小和颜色进行修改
				plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
			else:
				plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
	if(intFlag):
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
	else:
		plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
	plt.gca().set_xticks(tick_marks, minor=True)
	plt.gca().set_yticks(tick_marks, minor=True)
	plt.gca().xaxis.set_ticks_position('none')
	plt.gca().yaxis.set_ticks_position('none')
	plt.grid(True, which='minor', linestyle='-')
	plt.gcf().subplots_adjust(bottom=0.25)
	plt.title('Confusion Matrix with AID')
	plt.colorbar()
	xlocations = np.array(range(len(labels)))
	plt.xticks(xlocations, labels, rotation=90)
	plt.yticks(xlocations, labels)
	plt.ylabel('True Classes')
	plt.xlabel('Predict Classes')
	plt.savefig('confusion_matrix.jpg', dpi=600)
	plt.show()