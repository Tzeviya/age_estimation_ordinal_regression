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


import train
import train_coral

import model_orig
import vgg

import resnet
import model_vgg
import model_resnet
import model_vgg_coral
import model_resnet_coral
# import vgg_audio

import dataset_historical
import dataset_afad
import dataset_imdb_wiki
import dataset_utkface
import dataset_tidigits
import dataset_kaggle


parser = argparse.ArgumentParser(description='train ordinal regression')
parser.add_argument('--train', type=str, default='datasets/historical_color_dataset/train.txt', 
					help='location of data')
parser.add_argument('--val', type=str, default='datasets/historical_color_dataset/val.txt', 
					help='location of the validation wrd files')
parser.add_argument('--test', type=str, default='datasets/historical_color_dataset/test.txt', 
					help='location of the test wrd files')

parser.add_argument('--num_classes', type=int, default=5, help='num of classes')
parser.add_argument('--opt', type=str, default='adam', help='optimization method: adam || sgd')
parser.add_argument('--bn', action='store_true', help='add batchnorm to vgg net')

parser.add_argument('--compare_by_acc', action='store_true', help='save model with best validation accuracy rather than loss')
parser.add_argument('--compare_by_mae', action='store_true', help='save model with best validation MAE rather than loss')

parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate (or 0.001)')
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
parser.add_argument('--dropout', type=float,  default=0.0, help='dropout probability value')
parser.add_argument('--seed', type=int, default=1245, help='random seed')
parser.add_argument('--patience', type=int, default=10, help='patience')


parser.add_argument('--arc', type=str, default='VGG16', help='orig || VGG16 || VGG19 || resnet18 || resnet34 || resnet50 ')
parser.add_argument('--single_layer', action='store_true', help='g_r and g_c are a single layer')

parser.add_argument('--log_interval', type=int, default=100, help='log log interval')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--save_folder', type=str, default='models/', help='save folder')
parser.add_argument('--save_file', type=str, default='model.pth', help='save file')


parser.add_argument('--mfcc', action='store_true', help='use mfcc features instead of stft')
parser.add_argument('--pretrain', action='store_true', help='use pretrained model')


parser.add_argument('--historical', action='store_true', help='run historical color dataset dataset')
parser.add_argument('--afad', action='store_true', help='run AFAD dataset')
parser.add_argument('--imdb', action='store_true', help='run IMDB-WIKI dataset')
parser.add_argument('--utk', action='store_true', help='run UTKFace dataset')
parser.add_argument('--tidigits', action='store_true', help='run TIDIGITS dataset')
parser.add_argument('--kaggle', action='store_true', help='run kaggle dataset')

parser.add_argument('--small', action='store_true', help='run on small dataset for testing')

parser.add_argument('--coral', action='store_true', help='run CORAL algorithm')


args = parser.parse_args()
print(args)



args.cuda = args.cuda and torch.cuda.is_available()
# torch.manual_seed(args.seed)
# random.seed(args.seed)

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

# pdb.set_trace()
if args.afad:
	train_loader, val_loader, test_loader = \
				  dataset_afad.load_dataset('datasets/afad_dataset/train.txt', 'datasets/afad_dataset/val.txt', 'datasets/afad_dataset/test.txt', \
					batch_size=int(args.batch_size), device=args.cuda, small = args.small)

elif args.imdb:
	train_loader, val_loader, test_loader = \
				  dataset_imdb_wiki.load_dataset('datasets/imdb_wiki_dataset/train.txt', 'datasets/imdb_wiki_dataset/val.txt', 'datasets/imdb_wiki_dataset/test.txt', \
					train_age_range=None, test_age_range=None, batch_size=int(args.batch_size), device=args.cuda, small = args.small)

elif args.utk:
	train_loader, val_loader, test_loader = \
				  dataset_utkface.load_dataset('datasets/UTKFace/train.txt', 'datasets/UTKFace/val.txt', 'datasets/UTKFace/test.txt', \
					batch_size=int(args.batch_size), device=args.cuda, small = args.small)

elif args.tidigits:
	train_loader, val_loader, test_loader = \
				  dataset_tidigits.load_dataset('datasets/TIDIGITS/train.txt', 'datasets/TIDIGITS/val.txt', 'datasets/TIDIGITS/test.txt', \
					batch_size=int(args.batch_size), device=args.cuda, mfcc = args.mfcc, small = args.small)
elif args.kaggle:
	train_loader, val_loader, test_loader = \
				  dataset_kaggle.load_dataset('datasets/kaggle/train.txt', 'datasets/kaggle/val.txt', 'datasets/kaggle/test.txt', \
					batch_size=int(args.batch_size), device=args.cuda, small = args.small)

elif args.historical:
	  train_loader, val_loader, test_loader = \
				  dataset_kaggle.load_dataset('datasets/historical_color_dataset/train.txt', 'datasets/historical_color_dataset/val.txt', 'datasets/historical_color_dataset/test.txt', \
	
					batch_size=int(args.batch_size), device=args.cuda, small = args.small)
else:
	train_loader, val_loader, test_loader = \
				dataset_historical.load_dataset(args.train, args.val, args.test, batch_size=int(args.batch_size), device=args.cuda, small = args.small)


if args.tidigits: #TODO: or any other audio dataset
	input_layer = 1
	pretrained = False
	orig_num_classes = 30
else:
	input_layer = 3
	pretrained = args.pretrain
	orig_num_classes = 1000

if args.coral: #coral implementation

	if args.arc == 'orig':
		model = model_orig.MultiCNN(args.num_classes)
	elif 'VGG' in args.arc:
		if args.bn:
			this_arc = args.arc.lower() + '_bn'
		else:
			this_arc = args.arc.lower()

		chosen_model = getattr(vgg, this_arc)(input_layer, pretrained=pretrained, progress=pretrained, num_classes = orig_num_classes)
		tail_model = model_vgg_coral.NetTail(args.num_classes, orig_num_classes, chosen_model.classifier, args.single_layer)
		model = model_vgg_coral.Net(chosen_model, tail_model)
	elif 'resnet' in args.arc:
		chosen_model = getattr(resnet, args.arc)(input_layer, pretrained=pretrained, progress=pretrained)

		block_expansion = 1 #is of type resnet.BasicBlock 
		for m in chosen_model.modules(): 
			if isinstance(m, resnet.Bottleneck):
				block_expansion = 4 #is of type resnet.Bottleneck
				break

		tail_model = model_resnet_coral.NetTail(args.num_classes, args.single_layer, block_expansion)
		model = model_resnet_coral.Net(chosen_model, tail_model)



else:
	if args.arc == 'orig':
		model = model_orig.MultiCNN(args.num_classes)
	elif 'VGG' in args.arc:
		if args.bn:
			this_arc = args.arc.lower() + '_bn'
		else:
			this_arc = args.arc.lower()

		chosen_model = getattr(vgg, this_arc)(input_layer, pretrained=pretrained, progress=pretrained, num_classes = orig_num_classes)
		tail_model = model_vgg.NetTail(args.num_classes, orig_num_classes, chosen_model.classifier, args.single_layer)
		model = model_vgg.Net(chosen_model, tail_model)
	elif 'resnet' in args.arc:
		chosen_model = getattr(resnet, args.arc)(input_layer, pretrained=pretrained, progress=pretrained)

		block_expansion = 1 #is of type resnet.BasicBlock 
		for m in chosen_model.modules(): 
			if isinstance(m, resnet.Bottleneck):
				block_expansion = 4 #is of type resnet.Bottleneck
				break

		tail_model = model_resnet.NetTail(args.num_classes, args.single_layer, block_expansion)
		model = model_resnet.Net(chosen_model, tail_model)



if args.cuda:
	print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
	model = torch.nn.DataParallel(model).cuda()


print('==>optimizer: {} and learning rate: {}'.format(args.opt, args.lr))
if args.opt == 'sgd':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
elif args.opt == 'adagrad':
	optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
elif args.opt == 'adadelta':
	optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
elif args.opt == 'adam':
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
	raise RuntimeError("Invalid optim method: " + args.opt)


# pdb.set_trace()

best_valid_mae = np.inf
best_valid_loss = np.inf
best_valid_acc = 0.0
iteration = 0
epoch = 1

if args.coral:

	def cost_fn(logits, levels, imp = 1):
		# pdb.set_trace()
		val = (-torch.sum((F.logsigmoid(logits)*levels + (F.logsigmoid(logits) - logits)*(1-levels))*imp,
			   dim=1))
		return torch.mean(val)

	crossent_loss = cost_fn

else:
	crossent_loss = nn.CrossEntropyLoss() #This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

# pdb.set_trace()

# training with early stopping
while (epoch < args.epochs + 1) and (iteration < args.patience):
	# pdb.set_trace()
	if args.coral:
		train_coral.train(train_loader, model, optimizer, crossent_loss, epoch, device, args.log_interval)
		valid_loss, valid_acc, valid_mae = train_coral.test(val_loader, model, crossent_loss, device)
	else:
		train.train(train_loader, model, optimizer, crossent_loss, epoch, device, args.log_interval)
		valid_loss, valid_acc, valid_mae = train.test(val_loader, model, crossent_loss, device)
	print('validation loss: ', valid_loss)
	# pdb.set_trace()
	save_model_flag = True
	if args.compare_by_acc:
		if valid_acc < best_valid_acc:
			iteration += 1
			save_model_flag = False
			print('Accuracy was not improved, iteration {0}'.format(str(iteration)))
	elif args.compare_by_mae:
		if valid_mae > best_valid_mae:
			iteration += 1
			save_model_flag = False
			print('MAE was not improved, iteration {0}'.format(str(iteration)))
	else:
		if valid_loss > best_valid_loss:
			iteration += 1
			save_model_flag = False
			print('Loss was not improved, iteration {0}'.format(str(iteration)))

	if save_model_flag:
		print('Saving model...')
		iteration = 0
		best_valid_loss = valid_loss
		best_valid_acc = valid_acc
		best_valid_mae = valid_mae

		
		state = {
			'net': model.module.state_dict() if args.cuda else model.state_dict(), #correct if using "dataparallel"
			'acc': valid_acc,
			'loss': valid_loss,
			'epoch': epoch,
			}
		if not os.path.isdir(args.save_folder):
			os.mkdir(args.save_folder)
		torch.save(state, args.save_folder + '/' + args.save_file)
	epoch += 1


print('**********************************')
print('**********TESTING*****************')
if args.coral:
	test_loss, test_acc, test_mae = train_coral.test(test_loader, model, crossent_loss, device)
else:
	test_loss, test_acc, test_mae = train.test(test_loader, model, crossent_loss, device)
print('test loss: ', test_loss)