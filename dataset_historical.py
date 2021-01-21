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


def make_dataset_pointwise(data_path, mode, small = False):

	return_examples = []
	all_examples = {}
	
	data = open(data_path, 'r').readlines()

	for example in data:
		label = example.split('/')[-2]

		if not label in all_examples:
			all_examples[label] = []
		all_examples[label].append(example.strip('\n'))

	all_labels = list(all_examples.keys())
	print(all_labels)

	i = 0
	for example in data:
		if small and i == 1000:
			break
		return_examples.append(example.strip())
		i += 1

	return return_examples, all_labels

	# pdb.set_trace()
	


class HistoricalImages(data.Dataset): 
	def __init__(self, data_path, transform, mode, small = False):

		self.examples, self.labels = make_dataset_pointwise(data_path, mode, small) 
		self.transform = transform
		if len(self.examples) == 0:
			raise (RuntimeError("Found 0 files"))


	def __len__(self):
		return len(self.examples)


	def __getitem__(self, idx):


		# The images in
		# the historical image dataset are resized to 256 × 256 pixels.
		# For all the three deep learning methods, the image size of
		# the input layer is set to 224 × 224 3-channel pixels, and the
		# input images are cropped further at random positions during
		# the training phases for data augmentation. 

		example = self.examples[idx]

		img = Image.open(example)
		img.load()
		
		if self.transform:
			img = self.transform(img)

		label = os.path.basename(os.path.abspath(os.path.join(example, os.pardir)))
		label_idx = self.labels.index(label)

		return img, label_idx



def load_dataset(train_path, val_path, test_path, batch_size, device, small = False):

	random.seed(0)

	
	transform = transforms.Compose([transforms.Resize([256, 256]),
									transforms.RandomCrop(224),
									transforms.ToTensor()])

	
	train_dataset, val_dataset, test_dataset = [], [], []
	if train_path: train_dataset = HistoricalImages(train_path, transform, 'train', small)
	if val_path: val_dataset = HistoricalImages(val_path, transform, 'val', small)
	if test_path: test_dataset = HistoricalImages(test_path, transform, 'test', small)


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