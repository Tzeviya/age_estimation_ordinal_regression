
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


#original architecture used in paper

class MultiCNN(nn.Module):
	def __init__(self, num_classes, init_weights=False):
		super(MultiCNN, self).__init__()

		self.num_fcs = num_classes - 1

		self.conv1 = nn.Conv2d(3, 20, 5, stride = 1)
		self.conv2 = nn.Conv2d(20, 40, 7, stride = 1)
		self.conv3 = nn.Conv2d(40, 80, 11, stride = 1)

		self.maxpool = nn.MaxPool2d(2, 2)
		self.norm = nn.LocalResponseNorm(5) #TODO: should this be 5? https://caffe.berkeleyvision.org/tutorial/layers/lrn.html


		for i in range(self.num_fcs):
			setattr(self, "fc%d" % i, nn.Linear(80, 2))		

		if init_weights:
			self._initialize_weights()

	def forward(self, x):

		x = self.maxpool(self.norm(F.relu(self.conv1(x))))
		x = self.maxpool(self.norm(F.relu(self.conv2(x))))
		x = self.norm(F.relu(self.conv3(x)))
		x = x.squeeze()

		clf_outputs = {}
		for i in range(self.num_fcs):
			clf_outputs["fc%d" % i] = getattr(self, "fc%d" % i)(x)

		return clf_outputs
		


	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)



	@staticmethod
	def load_model(model, bn, num_classes, single_layer, arc, device):

		print('reimplement load_model')
		exit()

		checkpoint = torch.load(model, map_location=lambda storage, loc: storage)

		if arc == 'VGG16':
			if bn: g_h = vgg.vgg16_bn() 
			else: g_h = vgg.vgg16() 
		elif arc == 'VGG19':
			if bn: g_h = vgg.vgg19_bn() 
			else: g_h = vgg.vgg19()

		# pdb.set_trace()
		g_h.load_state_dict(checkpoint['g_h'])

		if 'g_c' in checkpoint:
			g_c = clssificiationNet(num_classes, g_h.classifier, single_layer)
			g_c.load_state_dict(checkpoint['g_c'])
		else:
			g_c = None

	
		# print('=======>change 1000 to 30 for tidigits')
		if 'g_r' in checkpoint:
			g_r = rankingNet(1, 1000, g_h.classifier, single_layer) #change '1000' to '30' for tidigits
			g_r.load_state_dict(checkpoint['g_r'])
		else:
			g_r = None

		model_ = SiameseNet(g_h, g_c, g_r)
		
		# model_.g_h.load_state_dict(checkpoint['g_h'])
		# model_.g_c.load_state_dict(checkpoint['g_c'])
		# model_.g_r.load_state_dict(checkpoint['g_r'])

		if 'theta' in checkpoint:
			list_of_thresholds = checkpoint['theta']
		else:
			list_of_thresholds = {}
			
		loss = checkpoint['loss']
		acc = checkpoint['acc']
		epoch = checkpoint['epoch']
		return model_, list_of_thresholds, loss, acc, epoch



