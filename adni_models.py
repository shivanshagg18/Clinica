# File to store ADNI model architectures

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
	def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, bias=False):
		padding = (kernel_size - 1) // 2
		super(ConvBNReLU, self).__init__(
			nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
			nn.BatchNorm3d(out_planes),
			nn.ReLU6(inplace=True)
		)

class ADNINetwork(nn.Module):
	def __init__(self):
		super(ADNINetwork, self).__init__()

		self.convInit()
		self.n_classes = 2
		self.linear1 = nn.Linear(256, 64, bias=True)
		self.linear2 = nn.Linear(64, self.n_classes, bias=True)

	def forward(self, x):
		
		out = self.convbnl1(x)
		out = self.convbnl2(out)
		out = self.convbnl3(out)
		out = self.convbnl4(out)

		out = F.avg_pool3d(out, 4)

		out = out.view(out.size(0), -1)
		out = self.linear1(out)
		out = self.linear2(out)

		if self.n_classes == 1:
			out = out.view(out.size(0))

		return out

	def convInit(self):
		self.convbnl1 = ConvBNReLU(1, 4, kernel_size=3, stride=2)
		self.convbnl2 = ConvBNReLU(4, 16, kernel_size=3, stride=2)
		self.convbnl3 = ConvBNReLU(16, 64, kernel_size=3, stride=2)
		self.convbnl4 = ConvBNReLU(64, 256, kernel_size=3, stride=2)