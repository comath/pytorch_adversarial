import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from tqdm import tqdm

from attackTemplate import BaseAttack

class GradientAttack(BaseAttack):
	def __init__(self, model, loss, epsilon, useLabels = False, pixel_range=(-1,1)):
		super(GradientAttack, self).__init__()
		self.model = model
		self.loss = loss
		self.epsilon = epsilon
		self.pixel_range = pixel_range
		self.useLabels = useLabels

	def sampleLabels(self,x):
		outputs = self.model(x)
		_, predicted = torch.max(outputs.data, 1)
		return predicted

	@property
	def usesLabels(self):
		return self.useLabels

	def forward(self,x,y_true = None):
		x_adv = x.requires_grad_()
		if y_true is None:
			y_true = self.sampleLabels(x)


		y = self.model.forward(x_adv)
		J = self.loss(y,y_true)

		if x_adv.grad is not None:
			x_adv.grad.data.fill_(0)

		x_grad = torch.autograd.grad(J, x_adv)[0]
		x_adv = x_adv + self.epsilon*x_grad
		#x_adv = torch.clamp(x_adv, self.pixel_range[0], self.pixel_range[1])

		return x_adv

class NormalizedGradientAttack(GradientAttack):
	def forward(self,x,y_true = None):
		x_adv = x.requires_grad_()
		if y_true is None:
			y_true = self.sampleLabels(x)

		y = self.model.forward(x_adv)
		J = self.loss(y,y_true)

		if x_adv.grad is not None:
			x_adv.grad.data.fill_(0)

		x_grad = torch.autograd.grad(J, x_adv)
		x_grad = x_grad[0]

		x_norm = x_grad.view(x_grad.size()[0], -1)
		x_norm = F.normalize(x_norm)
		x_norm = x_norm.view(x_grad.size())

		x_adv = x_adv + self.epsilon*x_norm
		#x_adv = torch.clamp(x_adv, self.pixel_range[0], self.pixel_range[1])

		return x_adv


class GradientSignAttack(GradientAttack):
	def forward(self,x,y_true = None):
		x_adv = x.requires_grad_()
		if y_true is None:
			y_true = self.sampleLabels(x)
		

		y = self.model.forward(x_adv)
		J = self.loss(y,y_true)

		if x_adv.grad is not None:
			x_adv.grad.data.fill_(0)

		x_grad = torch.autograd.grad(J, x_adv)[0]
		x_adv = x + self.epsilon*x_grad.sign_()
		# Defaults to 0. Which is screwed up.
		#x_adv = torch.clamp(x_adv, self.pixel_range[0], self.pixel_range[1])
		return x_adv