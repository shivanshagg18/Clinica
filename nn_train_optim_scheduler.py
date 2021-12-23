# This file contains the training loop
# do_epoch is the most basic loop possible
# Add more complicated training in different functions and name accordingly

import torch
import numpy as np
import random
from tqdm import tqdm

from utils import prepare_batch

def do_epoch(model, dataloader, criterion, optim=None, device='cpu'):
	total_loss = 0
	total_accuracy = 0
	if optim is not None:
		model.train()
	else:
		model.eval()

	for batch_idx, batch in enumerate(tqdm(dataloader, leave=True)):
		x, y_true = prepare_batch(batch, device)
		y_pred = model(x)
		loss = criterion(y_pred, y_true)

		if optim is not None:
			optim.zero_grad()
			loss.backward()
			optim.step()

		total_loss += loss.item()
		total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()

	mean_loss = total_loss / len(dataloader)
	mean_accuracy = total_accuracy / len(dataloader)

	return mean_loss, mean_accuracy
