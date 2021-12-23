# This is the main training files in this repo
# can also be used for testing (just set epochs to 0; to skip training) 
# Add more command line arguments as and when necessary

import enum
import time
import random
import multiprocessing
from pathlib import Path
import glob
import pdb
import argparse

import torch
import torchvision
import torchio as tio
import torch.nn.functional as F

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from IPython import display
from tqdm import tqdm

from utils import manual_seed
from nn_train_optim_scheduler import do_epoch

def main(args):
	
	manual_seed(args.seed)

	# num_workers = multiprocessing.cpu_count()
	# plt.rcParams['figure.figsize'] = 12, 6

	print('Last run on', time.ctime())
	print('TorchIO version:', tio.__version__)

	adnimerge_dict_file = args.adnimerge_dict_file_path
	adnimerge_file = args.adnimerge_file_path
	adnimerge_dict = pd.read_csv(adnimerge_dict_file)
	adnimerge = pd.read_csv(adnimerge_file)

	clinica_CAPS_path = args.clinica_CAPS_path
	all_files = glob.glob(clinica_CAPS_path+"/*/*/*/*/*/*/*/*.nii.gz")

	subjects = []

	for file in all_files:
		id_str = file.split("/")[-3]
		ptid = id_str[8:11]+"_S_"+id_str[12:16]
		viscode = id_str.split("-")[-1].lower()
		print(id_str)

		specific_row = adnimerge.loc[(adnimerge['PTID']==ptid) & (adnimerge['VISCODE']==viscode)]
		if len(specific_row)==0:
			specific_row = adnimerge.loc[(adnimerge['PTID']==ptid) & (adnimerge['VISCODE']=='bl')]

		assert len(specific_row) == 1
		dx = specific_row['DX_bl'].item()

		if dx!='AD':
			correct_dx = 0
		else:
			correct_dx = 1
		print(correct_dx)

		subject = tio.Subject(
					mri=tio.ScalarImage(file),
					diagbl=correct_dx
					)
		subjects.append(subject)

	dataset = tio.SubjectsDataset(subjects)
	print('Dataset size:', len(dataset), 'subjects')

	one_subject = dataset[0]
	# one_subject.plot()

	print("##### First subject ######")
	print(one_subject)
	print(one_subject.mri)
	print(one_subject.diagbl)
	print("##### First subject ######")

	num_subjects = len(dataset)
	num_training_subjects = int(args.split_ratio * num_subjects)
	num_validation_subjects = num_subjects - num_training_subjects

	num_split_subjects = num_training_subjects, num_validation_subjects
	training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

	training_set = tio.SubjectsDataset(training_subjects, transform=None)

	validation_set = tio.SubjectsDataset(validation_subjects, transform=None)

	print('Training set:', len(training_set), 'subjects')
	print('Validation set:', len(validation_set), 'subjects')

	train_loader = torch.utils.data.DataLoader(
		training_set,
		batch_size=args.train_bs,
		shuffle=True,
		num_workers=args.num_workers,
	)
	val_loader = torch.utils.data.DataLoader(
		validation_set,
		batch_size=args.val_bs,
		num_workers=args.num_workers,
	)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	exec("from adni_models import %s" % args.model)
	model = eval(args.model)().to(device)
	print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

	if args.optimizer == 'sgd':
		optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
	elif args.optimizer == 'adam':
		optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	else:
		error('unknown optimizer')

	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=200)
	criterion = torch.nn.CrossEntropyLoss()

	best_accuracy = 0
	for epoch in range(args.train_epochs):
		train_loss, train_accuracy = do_epoch(model, train_loader, criterion, optim, device)

		with torch.no_grad():
			val_loss, val_accuracy = do_epoch(model, val_loader, criterion, None, device)

		tqdm.write(f'{args.model} EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
				   f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

		if val_accuracy > best_accuracy:
		   print('Saving best model...')
		   best_accuracy = val_accuracy
		   torch.save(model.state_dict(), args.output_model_path+'best_model.pt')

		lr_scheduler.step()

	with torch.no_grad():
		fin_loss, fin_accuracy = do_epoch(model, val_loader, criterion, None, device)

	tqdm.write(f'{args.model} final_val_loss={fin_loss:.4f}, final_val_accuracy={fin_accuracy:.4f}')

	print('Saving final model...')
	torch.save(model.state_dict(), args.output_model_path+'final_model.pt')


if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser(description='Basic Training/Testing ADNI models')

	# Seed and path variables
	arg_parser.add_argument('--seed', type=int, default=0, help="Fix seeds for reproducibility")
	arg_parser.add_argument('--adnimerge_file_path', type=str, default='./ADNIMERGE.csv')
	arg_parser.add_argument('--adnimerge_dict_file_path', type=str, default='./ADNIMERGE_DICT.csv')
	arg_parser.add_argument('--clinica_CAPS_path', type=str, default="../clinica_CAPS/", help="Path to clinica fressurfer output CAPS folder")
	arg_parser.add_argument('--output_model_path', type=str, default="./trained_models/", help="Path to save ADNI trained models")

	# Model and optimizer, add other parameters as necessary for optimizers, name accordingly
	arg_parser.add_argument('--model', type=str, default='ADNINetwork', help='model for training/testing')
	arg_parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer from Pytorch', choices=['sgd','adam'])
	arg_parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')

	# Data loading arguments, add and name accordingly
	arg_parser.add_argument('--train_bs', type=int, default=2, help='Training batch size')
	arg_parser.add_argument('--val_bs', type=int, default=2, help='Validation/Testing batch size')

	arg_parser.add_argument('--train_epochs', type=int, default=10, help='Number of epochs to train for')
	arg_parser.add_argument('--split_ratio', type=float, default=0.8, help='Ratio to split dataset between training and testing')
	arg_parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloading')
	
	args = arg_parser.parse_args()

	main(args)
