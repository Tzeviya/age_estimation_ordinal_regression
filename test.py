import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import random
import torch
import time
import os
import pdb

import model_vgg
import model_resnet

import model_vgg_coral
import model_resnet_coral

import vgg
import resnet

import dataset_historical
# import dataset_historical_pointwise
import dataset_afad
import dataset_imdb_wiki
import dataset_utkface
import dataset_tidigits
import dataset_gov
import dataset_fisher
import train
import train_coral


parser = argparse.ArgumentParser(description='test ordinal regression')

parser.add_argument('--test', type=str, default='datasets/historical_color_dataset/test.txt', 
					help='location of the test wrd files')

parser.add_argument('--num_classes', type=int, default=5, help='num of classes')
parser.add_argument('--bn', action='store_true', help='add batchnorm to vgg net')

parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
parser.add_argument('--dropout', type=float,  default=0.0, help='dropout probability value')
parser.add_argument('--seed', type=int, default=1245, help='random seed')
parser.add_argument('--patience', type=int, default=10, help='patience')
parser.add_argument('--arc', type=str, default='VGG16', help='VGG16 || VGG19 ')

parser.add_argument('--single_layer', action='store_true', help='g_r and g_c are a single layer')
parser.add_argument('--mfcc', action='store_true', help='use mfcc features instead of stft')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--model', type=str, default='model/model.pth', help='save file')

parser.add_argument('--historical', action='store_true', help='run historical color dataset dataset')
parser.add_argument('--afad', action='store_true', help='run AFAD dataset')
parser.add_argument('--imdb', action='store_true', help='run IMDB-WIKI dataset')
parser.add_argument('--utk', action='store_true', help='run UTKFace dataset')
parser.add_argument('--tidigits', action='store_true', help='run TIDIGITS dataset')
parser.add_argument('--gov', action='store_true', help='run GOV dataset')
parser.add_argument('--fisher', action='store_true', help='run FISHER dataset')

parser.add_argument('--coral', action='store_true', help='run CORAL algorithm')

parser.add_argument('--small', action='store_true', help='run on small dataset for testing')

# parser.add_argument('--imdb_age_range', action='store_true', help='run train on reduced set ot labels')

args = parser.parse_args()
print(args)


args.cuda = args.cuda and torch.cuda.is_available()


#init seeds from SO
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)


if args.cuda:
	torch.cuda.manual_seed(args.seed)
	device = 'cuda'
else:
	device = 'cpu'


if args.coral: #coral implementation
	if 'VGG' in args.arc:
		net, loss, acc, epoch = model_vgg_coral.Net.load_model(args.model, args.bn, args.num_classes, args.single_layer, args.arc)

	elif 'resnet' in args.arc:
		net, loss, acc, epoch = model_resnet_coral.Net.load_model(args.model, args.bn, args.num_classes, args.single_layer, args.arc)


else:
	if args.arc == 'orig':
		model = model_orig.MultiCNN(args.num_classes)
	elif 'VGG' in args.arc:
		net, loss, acc, epoch = model_vgg.Net.load_model(args.model, args.bn, args.num_classes, args.single_layer, args.arc)

		
	elif 'resnet' in args.arc:
		net, loss, acc, epoch = model_resnet.Net.load_model(args.model, args.bn, args.num_classes, args.single_layer, args.arc)


if args.cuda:
	print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
	net = torch.nn.DataParallel(net).cuda()

if args.afad:
	train_loader, val_loader, test_loader = \
				  dataset_afad.load_dataset(None, None, 'datasets/afad_dataset/test.txt', \
					batch_size=int(args.batch_size), device=args.cuda, small = args.small)

elif args.imdb:
	if args.imdb_age_range:
		train_loader, val_loader, test_loader = \
					  dataset_imdb_wiki.load_dataset(None, None, 'datasets/imdb_wiki_dataset/test.txt', \
						train_age_range=(16,49), test_age_range=(50,59), batch_size=int(args.batch_size), device=args.cuda, small = args.small)

	else:
		train_loader, val_loader, test_loader = \
					  dataset_imdb_wiki.load_dataset(None, None, 'datasets/imdb_wiki_dataset/test.txt', \
						train_age_range=None, test_age_range=None, batch_size=int(args.batch_size), device=args.cuda, small = args.small)
		

elif args.utk:
	train_loader, val_loader, test_loader = \
				  dataset_utkface.load_dataset(None, None, 'datasets/UTKFace/test.txt', \
					batch_size=int(args.batch_size), device=args.cuda, small = args.small)

elif args.tidigits:
	train_loader, val_loader, test_loader = \
				  dataset_tidigits.load_dataset(None, None, 'datasets/TIDIGITS/test.txt', \
					batch_size=int(args.batch_size), device=args.cuda, mfcc = args.mfcc, small = args.small)

elif args.historical:
	train_loader, val_loader, test_loader = \
				dataset_historical.load_dataset(None, None, 'datasets/historical_color_dataset/test.txt', \
					batch_size=int(args.batch_size), device=args.cuda, small = args.small)
else:
	
	train_loader, val_loader, test_loader = \
			dataset_historical.load_dataset(None, None, args.test, batch_size=int(args.batch_size), device=args.cuda, small = args.small)


if args.coral:

	def cost_fn(logits, levels, imp = 1):
		# pdb.set_trace()
		val = (-torch.sum((F.logsigmoid(logits)*levels + (F.logsigmoid(logits) - logits)*(1-levels))*imp,
			   dim=1))
		return torch.mean(val)

	crossent_loss = cost_fn

else:
	crossent_loss = nn.CrossEntropyLoss() #This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.



if args.coral:
	test_loss, test_acc, test_mae = train_coral.test(test_loader, net, crossent_loss, device)
else:
	test_loss, test_acc, test_mae = train.test(test_loader, net, crossent_loss, device)

print('pointwise test loss: ', test_loss)

