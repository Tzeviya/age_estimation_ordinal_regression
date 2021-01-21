import torch
import torch.nn.functional as F
import pdb

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import shutil
import os

def train(train_loader, model, optimizer, crossent_loss, epoch, device, log_interval, print_progress=True):
	model.train()
	global_epoch_loss = 0

	# for param in model.parameters():
		#param.requires_grad = False

	for batch_idx, (example, label_idx) in enumerate(train_loader):
		
		optimizer.zero_grad()
		example, label_idx = example.to(device), label_idx.to(device)
		batch_size = example.shape[0]

		logits, probas = model(example)

		target = []
		num_classifiers = logits.shape[1]
		for this_label in label_idx:
			target.append(torch.Tensor([1 if i < this_label else 0 for i in range(num_classifiers) ]))
		target = torch.stack(target).to(device)

		# pdb.set_trace()
		loss = crossent_loss(logits, target)

		loss.backward()
		optimizer.step()
		
		global_epoch_loss += loss.item()
		if print_progress:
			if batch_idx % log_interval == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(example), len(train_loader.dataset), 100.
					* batch_idx / len(train_loader), loss))
	return global_epoch_loss / len(train_loader.dataset)

def test(test_loader, model, crossent_loss, device):
	with torch.no_grad():
		model.eval()
		total_loss = 0
		all_tp, all_ex, all_mae = 0, 0, 0 

		for batch_idx, (example, label_idx) in enumerate(test_loader):


			
			example, label_idx = example.to(device), label_idx.to(device)
			batch_size = example.shape[0]

			logits, probas = model(example)

			target = []
			num_classifiers = logits.shape[1]
			for this_label in label_idx:
				target.append(torch.Tensor([1 if i < this_label else 0 for i in range(num_classifiers) ]))
			target = torch.stack(target).to(device)

			loss = crossent_loss(logits, target)
			# pdb.set_trace()

			tp, abs_err = ranking_acc_mae(probas, label_idx, device)
			all_tp += tp
			all_mae += abs_err
			all_ex += len(label_idx)


			total_loss += loss.item()

		
		acc = all_tp / all_ex
		mae = all_mae/all_ex
		print('Accuracy: ', str(acc))
		print('MAE: ',str(mae))
		# print('*******not sure about the plus one in inference*********')


		return total_loss/len(test_loader.dataset), acc, mae



def ranking_acc_mae(probas, label_idx, device):

	# pdb.set_trace()

	predict_levels = probas > 0.5
	predicted_labels = torch.sum(predict_levels, dim=1)
	mae = torch.sum(torch.abs(predicted_labels - label_idx)).item()
	
	true_positives = sum((predicted_labels == label_idx).int()).item()

	return true_positives, mae
		


	# all_predictions = []
	# for idx, (key, value) in enumerate(clf_outputs.items()):
	# 	value = F.softmax(value, dim = 1)
	# 	prediction = torch.max(value, dim = 1)[1]
	# 	all_predictions.append(prediction)

	# all_predictions = torch.stack(all_predictions)

	# # pdb.set_trace()

	# y_hat = []
	# for ex in range(all_predictions.shape[1]):
	# 	ex_predictions = all_predictions[:, ex]
	# 	if plus_one: #I think I don't need to add plus one because my classes are zero indexed
	# 		y_hat.append(1 + sum(ex_predictions).item())
	# 	else:
	# 		y_hat.append(sum(ex_predictions).item())
	
	# y_hat = torch.Tensor(y_hat).to(device)
	# # pdb.set_trace()

	# true_positives = sum((y_hat == label_idx).int()).item()

	# abs_err = torch.abs(y_hat - label_idx)
	# sum_abs_err = sum(abs_err).item()
	
	# return true_positives, sum_abs_err



