import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import soundfile
import librosa
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

	# pdb.set_trace()
	for example in data:
		
		label, path = example.split('\t')
		label = int(label)

		if not label in all_examples:
			all_examples[label] = []
		all_examples[label].append(path.strip('\n'))

	all_labels = sorted(list(all_examples.keys()))
	print(all_labels) #[7, 8, 9, 10, 11, 12, 13, 15, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 48, 59]

	for l in all_labels:
		print(l, len(all_examples[l]))
	# exit()


	pic_dataset_dict = {} #path -> pic

	for example in data:

		age = int(example.split('\t')[0])

		if not age in all_labels:
			continue


		example = example.split('\t')[1].strip()

		if not example in all_examples[age]:
			continue
			
		pic_dataset_dict[example] = [example, age]

		return_examples.append(example)

	if small:
		return return_examples[:10], all_labels, pic_dataset_dict


	return return_examples, all_labels, pic_dataset_dict


def spect_loader(path, window_size=.02, window_stride=.01, window='hamming', normalize=True, max_len=51, mfcc=False):
	y, sr = soundfile.read(path)  
	y_original_len = len(y)

	try:
		n_fft = int(sr * window_size)
	except:
		print(path)
	
	win_length = n_fft
	hop_length = int(sr * window_stride)

	# STFT
	if mfcc == False:
		D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
					 win_length=win_length, window=window)
		spect, phase = librosa.magphase(D)
		spect = np.log1p(spect)
	else:
		mfcc_f = librosa.feature.mfcc(y, sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length, win_length=win_length)#, n_mels=120) #win_length=win_length, window=window
		mfcc_delta = librosa.feature.delta(mfcc_f)
		mfcc_delta2 = librosa.feature.delta(mfcc_f, order=2)
		spect = np.concatenate((mfcc_f, mfcc_delta, mfcc_delta2))

	# pdb.set_trace()
	real_features_len = spect.shape[1]
	# make all spects with the same dims
	if spect.shape[1] < max_len:
		pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
		spect = np.hstack((spect, pad))
	elif spect.shape[1] > max_len:
		spect = spect[:, :max_len]

	if mfcc:
		number_of_features = 39
	else: #sftf
		# number_of_features = 161
		number_of_features = 201


	if spect.shape[0] < number_of_features:
		pad = np.zeros((number_of_features - spect.shape[0], spect.shape[1]))
		spect = np.vstack((spect, pad))
	elif spect.shape[0] > number_of_features:
		spect = spect[:number_of_features, :]
	spect = np.resize(spect, (spect.shape[0], spect.shape[1]))
	spect = torch.FloatTensor(spect)

	spect = spect.T

	
	# z-score normalization
	if normalize:
		if not mfcc:
			mean = spect.mean()
			std = spect.std()
			if std != 0:
				spect.add_(-mean)
				spect.div_(std)

		else:
			mean = spect.mean(0)
			# print(mean.shape)
			spect.add_(-mean)
	
	spect = spect.T #[features, len]

	return spect.unsqueeze(0), len(y)

class TIDIGITS(data.Dataset): 
	def __init__(self, data_path, mfcc, mode, small = False):


		self.examples, self.labels, self.pic_dataset_dict = make_dataset_pointwise(data_path, mode, small) 
		self.mfcc = mfcc

		if len(self.examples) == 0:
			raise (RuntimeError("Found 0 files"))


	def __len__(self):
		return len(self.examples)


	def __getitem__(self, idx):

		example = self.examples[idx]

		spect, len_y = spect_loader(example, mfcc = self.mfcc) #max_len = self.spect_options.max_len,

		label = self.pic_dataset_dict[example][1] 
		label_idx = self.labels.index(label)

		return spect, label_idx

	


def load_dataset(train_path, val_path, test_path, batch_size, device, mfcc, small = False):

	random.seed(0)

	
	transform = None
	train_dataset, val_dataset, test_dataset = [], [], []
	if train_path: train_dataset = TIDIGITS(train_path, mfcc, 'train', small)
	if val_path: val_dataset = TIDIGITS(val_path, mfcc, 'val', small)
	if test_path: test_dataset = TIDIGITS(test_path, mfcc, 'test', small)

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