# File for uitlity functions

import torch
import torchio as tio
import numpy as np
import random
# from nilearn import plotting

def prepare_batch(batch, device):
	inputs = batch['mri'][tio.DATA].to(device=device, dtype=torch.float)
	targets = batch['diagbl'].to(device=device)
	return inputs, targets

def manual_seed(seed):
	print("Setting seeds to: ", seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def plot_group_maps(map_path):    
	map_file_name = map_path[:-7]
	plotting.plot_stat_map(stat_map_img=map_path, output_file=map_file_name+".png", threshold='auto')