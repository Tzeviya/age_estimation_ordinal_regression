import torch
import torch.nn as nn
import torch.nn.functional as F

import vgg
import resnet

import pdb



class NetTail(nn.Module):
	def __init__(self, num_classes, single_layer, block_expansion, init_weights=False):
		super(NetTail, self).__init__()

		self.num_fcs = num_classes - 1

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		if single_layer:

			self.classifier = []
			for i in range(self.num_fcs):
				setattr(self, "fc%d" % i, nn.Linear(512 * block_expansion, 2))	

			self.new_init()


		else:
			#check dims
			self.classifier = nn.Sequential(
				nn.Linear(512 * block_expansion, 4096),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(4096, 4096),
				nn.ReLU(True),
				nn.Dropout(),
				# nn.Linear(4096, num_classes),
			)


			for i in range(self.num_fcs):
				setattr(self, "fc%d" % i, nn.Linear(4096, 2))
			self.new_init()


		
			if init_weights:
				self._initialize_weights()

	def forward(self, x):
		x = self.avgpool(x)
		x = torch.flatten(x, 1)

		if not self.classifier == []:
			x = self.classifier(x)

		clf_outputs = {}
		for i in range(self.num_fcs):
			clf_outputs["fc%d" % i] = getattr(self, "fc%d" % i)(x)
		
		return clf_outputs

		return x

	def new_init(self):

		for i in range(self.num_fcs):
			layer = getattr(self, "fc%d" % i)

			nn.init.normal_(layer.weight, 0, 0.01)
			nn.init.constant_(layer.bias, 0)

			# nn.init.normal_(self.new_linear.weight, 0, 0.01)
			# nn.init.constant_(self.new_linear.bias, 0)

	def _initialize_weights(self):

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)



class Net(nn.Module):
	def __init__(self, vgg_net, tail_net):
		super(Net, self).__init__()

		self.g_h = vgg_net
		self.g_t = tail_net
		
	def forward(self, x):
		x = self.g_h(x)
		x = self.g_t(x)

		return x


	@staticmethod
	def load_model(model, bn, num_classes, single_layer, arc):

		if 'tidigits' in model:
			input_layer = 1
			orig_num_classes = 30
		else:
			input_layer = 3
			orig_num_classes = 1000

		
		# pdb.set_trace()
		checkpoint = torch.load(model, map_location=lambda storage, loc: storage)
		chosen_model = getattr(resnet, arc)(input_layer, pretrained=False, progress=False)

		block_expansion = 1 #is of type resnet.BasicBlock 
		for m in chosen_model.modules(): 
			if isinstance(m, resnet.Bottleneck):
				block_expansion = 4 #is of type resnet.Bottleneck
				break


		tail_model = NetTail(num_classes, single_layer, block_expansion)
		model = Net(chosen_model, tail_model)
		model.load_state_dict(checkpoint['net'])
			
		loss = checkpoint['loss']
		acc = checkpoint['acc']
		epoch = checkpoint['epoch']
		return model, loss, acc, epoch