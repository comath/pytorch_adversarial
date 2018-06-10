import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from tqdm import tqdm

from attackTemplate import BaseAttack

class GradientAttack(BaseAttack):
	def __init__(self, model, loss, epsilon, n = 1, useLabels = False):
		super(GradientAttack, self).__init__()
		self.model = model
		self.loss = loss
		self.epsilon = epsilon
		self.n = n
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

		for i in range(self.n):
			y = self.model.forward(x_adv)
			J = self.loss(y,y_true)

			if x_adv.grad is not None:
				x_adv.grad.data.fill_(0)

			x_grad = torch.autograd.grad(J, x_adv, allow_unused=True)[0]
			x_adv = x_adv + self.epsilon*x_grad

		return x_adv

class NormalizedGradientAttack(GradientAttack):
	def forward(self,x,y_true = None):
		x_adv = x.requires_grad_()
		if y_true is None:
			y_true = self.sampleLabels(x)

		for i in range(self.n):
			y = self.model.forward(x_adv)
			J = self.loss(y,y_true)

			if x_adv.grad is not None:
				x_adv.grad.data.fill_(0)

			x_grad = torch.autograd.grad(J, x_adv, allow_unused=True)[0]
			x_grad = x_grad/torch.norm(x_grad,2,0, keepdim=True)
			x_adv = x + self.epsilon*x_grad
		return x_adv


class GradientSignAttack(GradientAttack):
	def forward(self,x,y_true = None):
		x_adv = x.requires_grad_()
		if y_true is None:
			y_true = self.sampleLabels(x)
		
		for i in range(self.n):
			y = self.model.forward(x_adv)
			J = self.loss(y,y_true)

			if x_adv.grad is not None:
				x_adv.grad.data.fill_(0)

			x_grad = torch.autograd.grad(J, x_adv)[0]
			x_adv = x + self.epsilon*x_grad.sign_()
			x_adv = torch.clamp(x_adv, -1, 1)

		return x_adv


if __name__ == '__main__':
	from cifar10 import CIFAR10ResNet,residual
	from datasets import CIFAR10
	model = torch.load("cifarnetbn.pickle")
	dataset = CIFAR10()
	batch_size = 400

	loader = dataset.training(batch_size)
	if torch.cuda.is_available():
		print("Using GPU 0")
		device = torch.device("cuda:0")
	else:
		print("No GPU, using CPU")
		device = torch.device("cpu")
	cpu = torch.device("cpu")
	criterion = nn.CrossEntropyLoss()
	model.to(device)

	gs = GradientSignAttack(model,criterion,0.07)

	dataiter = iter(dataset.testing(batch_size))
	images, labels = dataiter.next()
	images = images.cuda()
	gs.visualize(images,model,filename="fgsm_0-07.png", diff_multiply = 1)

	successRate = gs.test(model,dataset.testing(batch_size))

	print("The success rate on a ResNet CIFAR with FGSM eps 0.07 is " % (successRate,))

