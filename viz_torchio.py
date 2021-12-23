# This is the main vizualization file in this repo
# Used to load data, trained model and visualize as required
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

from nilearn import plotting
import nibabel as nib

from utils import manual_seed, prepare_batch
from nn_train_optim_scheduler import do_epoch
from visualize_methods import VanillaBackProp

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
	all_files = glob.glob(clinica_CAPS_path+"/*/*/*/*/*/*/*/brain_Registered.nii.gz")

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

	training_set = tio.SubjectsDataset(
		training_subjects, transform=None)

	validation_set = tio.SubjectsDataset(
		validation_subjects, transform=None)

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

	interpreter = VanillaBackProp(model, device)

	cum_maps = [0] 

	for batch_idx, batch in enumerate(tqdm(val_loader, leave=True)):
		images, y_true = prepare_batch(batch, device)

		map_pt = interpreter.generate_gradients(images, args.viz_target_node)
		for i in range(images.shape[0]):
			cum_maps[0] += map_pt[i]
			if args.save_individual:
				torch.save(map_pt[i], args.output_viz_results_path+"/"+str(batch_idx)+"_"+str(i)+"_indi.pt")

	mode_map = cum_maps[0]/len(validation_set)
	torch.save(mode_map, args.output_viz_results_path+"/full.pt")

	grp_nifti_img = nib.Nifti1Image(mode_map.numpy()[0], np.eye(4))
	# plotting.plot_stat_map(stat_map_img=grp_nifti_img, output_file="full.png", cut_coords=(-50, 14), display_mode="yz", threshold=10**-3)
	plotting.plot_stat_map(stat_map_img=grp_nifti_img, output_file=args.output_viz_results_path+"/full.png", threshold='auto')


if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser(description='Basic Training/Testing ADNI models')

	# Seed and path variables
	arg_parser.add_argument('--seed', type=int, default=0, help="Fix seeds for reproducibility")
	arg_parser.add_argument('--adnimerge_file_path', type=str, default='../ADNIdata_Oct20_from_euler/adni_clinical_data/ADNIMERGE.csv')
	arg_parser.add_argument('--adnimerge_dict_file_path', type=str, default='../ADNIdata_Oct20_from_euler/adni_clinical_data/ADNIMERGE_DICT.csv')
	arg_parser.add_argument('--clinica_CAPS_path', type=str, default="../clinica_CAPS/", help="Path to clinica fressurfer output CAPS folder")
	arg_parser.add_argument('--output_model_path', type=str, default="./trained_models/", help="Path to save ADNI trained models")
	arg_parser.add_argument('--output_viz_results_path', type=str, default="./plots/", help="Path to save visualization for trained models")

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

	arg_parser.add_argument('--viz_target_node', type=int, default=1, help='Output node for which we want to visualize')
	arg_parser.add_argument('--save_individual', default=False, action='store_true')
	
	args = arg_parser.parse_args()

	main(args)
