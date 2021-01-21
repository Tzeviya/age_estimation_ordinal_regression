import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import itertools  
import string
import random
import glob
import os
import pdb


def make_dataset_loading_data(data_path, mode, small = False):

	return_examples = []
	all_examples = {}
	
	data = open(data_path, 'r').readlines()

	# pdb.set_trace()
	for example in data:
		label = int(example.split('/')[-3])

		if not label in all_examples:
			all_examples[label] = []
		all_examples[label].append(example.strip('\n'))

	all_labels = sorted(list(all_examples.keys()))
	print(all_labels)

	for l in all_labels:
		print(l, len(all_examples[l]))

	
	pic_dataset_dict = {} #path -> pic
	
	i = 0
	for example in data:

		if small and i == 1000:
			break

		label = int(example.strip().split('/')[-3])

		if not label in all_labels:
			continue

		if not example.strip() in all_examples[label]:
			continue

		return_examples.append(example.strip())
		example = example.strip()
		this_img = Image.open(example)
		this_img.load()
		pic_dataset_dict[example] = this_img
		i += 1

	return return_examples, all_labels, pic_dataset_dict
	


class AFADImages(data.Dataset): 
	def __init__(self, data_path, transform, mode, small = False):

		self.examples, self.labels, self.pic_dataset_dict = make_dataset_loading_data(data_path, mode, small) 
		self.transform = transform

		if len(self.examples) == 0:
			raise (RuntimeError("Found 0 files"))


	def __len__(self):
		return len(self.examples)


	def __getitem__(self, idx):

		
		example = self.examples[idx]
		# crop_size = 224 #224 x 224

		if self.pic_dataset_dict == {}:
			# pdb.set_trace()
			img = Image.open(example)
			img.load()

		else:
			img = self.pic_dataset_dict[example]
		

		# pdb.set_trace()
		if self.transform:
			data = self.transform(img)

		# import matplotlib.pyplot as plt
		label = example.split('/')[-3] #os.path.basename(os.path.abspath(os.path.join(neg_example, os.pardir)))
		label_idx = self.labels.index(int(label))

		return data, label_idx
	


def load_dataset(train_path, val_path, test_path, batch_size, device, small = False):

	random.seed(0)

	
	print('========pics cropped to size 224x224========')
	transform = transforms.Compose([transforms.Resize([256, 256]),
									transforms.RandomCrop(224),
									transforms.ToTensor()])

	
	train_dataset, val_dataset, test_dataset = [], [], []
	if train_path: train_dataset = AFADImages(train_path, transform, 'train', small)
	if val_path: val_dataset = AFADImages(val_path, transform, 'val', small)
	if test_path: test_dataset = AFADImages(test_path, transform, 'test', small)

	print(f"==> train - {len(train_dataset)} examples, val - {len(val_dataset)} examples, test - {len(test_dataset)} examples")

	if device == 'cpu': is_cuda = False 
	else: is_cuda = True

	train_loader, val_loader, test_loader = None, None, None
	
	if train_path:
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
			num_workers=20, pin_memory=is_cuda, sampler=None)#, collate_fn=PadCollate()) 
	
	if val_path:
		val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=None,
			num_workers=20, pin_memory=is_cuda, sampler=None)#, collate_fn=PadCollate()) 

	if test_path:
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=None,
			num_workers=20, pin_memory=is_cuda, sampler=None)#, collate_fn=PadCollate()) 

	return train_loader, val_loader, test_loader 