import torch
import torch.nn as nn
import torch.nn.functional as F
import vgg

import pdb

class NetTail(nn.Module):
	def __init__(self, num_classes, orig_num_classes, vgg_classifier, single_layer, init_weights=False):
		super(NetTail, self).__init__()

		self.num_fcs = num_classes - 1

		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		# self.classifier = nn.Sequential(
		# 	nn.Linear(512 * 7 * 7, 512),
		# 	nn.ReLU(True),
		# 	nn.Dropout(),
		# 	nn.Linear(512, 512),
		# 	nn.ReLU(True),
		# 	nn.Dropout(),
		# 	nn.Linear(512, num_classes),
		# )
		# pdb.set_trace()

		if single_layer:

			self.classifier = []
			for i in range(self.num_fcs):
				setattr(self, "fc%d" % i, nn.Linear(512 * 7 * 7, 2))	

			self.new_init()


		else:
			self.classifier = nn.Sequential(
				nn.Linear(512 * 7 * 7, 4096),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(4096, 4096),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(4096, orig_num_classes),
			)

			
			#copying weights from Imagenet
			self.classifier.load_state_dict(vgg_classifier.state_dict())
			self.classifier = nn.Sequential(self.classifier[:-1]) #remove the last layer

			for i in range(self.num_fcs):
				setattr(self, "fc%d" % i, nn.Linear(4096, 2))
			self.new_init()
			
		
			if init_weights:
				self._initialize_weights()

	def forward(self, x):
		#original version
		# x = self.avgpool(x)
		# x = torch.flatten(x, 1)
		# x = self.classifier(x)

		x = self.avgpool(x)
		x = torch.flatten(x, 1)

		if not self.classifier == []:
			x = self.classifier(x)

		clf_outputs = {}
		for i in range(self.num_fcs):
			clf_outputs["fc%d" % i] = getattr(self, "fc%d" % i)(x)
		
		return clf_outputs

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
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
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

		checkpoint = torch.load(model, map_location=lambda storage, loc: storage)


		if bn:
			this_arc = arc.lower() + '_bn'
		else:
			this_arc = arc.lower()

		g_h = getattr(vgg, this_arc)(input_layer, pretrained=False, progress=False, num_classes = orig_num_classes)

		# pdb.set_trace()
		tail_model = NetTail(num_classes, orig_num_classes, g_h.classifier, single_layer)
		model = Net(g_h, tail_model)
		model.load_state_dict(checkpoint['net'])

			
		loss = checkpoint['loss']
		acc = checkpoint['acc']
		epoch = checkpoint['epoch']
		return model, loss, acc, epoch




