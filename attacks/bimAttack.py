import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from tqdm import tqdm

from attackTemplate import BaseAttack

class ItteratedGradientAttack(BaseAttack):
	def __init__(self, model, loss, epsilon, num_steps, step_size, p = 2, useLabels = False, pixel_range=(-1,1)):
		super(ItteratedGradientAttack, self).__init__()
		self.model = model
		self.loss = loss
		self.epsilon = epsilon
		self.step_size = step_size
		self.pixel_range = pixel_range
		self.useLabels = useLabels

	def sampleLabels(self,x):
		outputs = self.model(x)
		_, predicted = torch.max(outputs.data, 1)
		return predicted

	@property
	def usesLabels(self):
		return self.useLabels

	def step(self,x_adv,y_true):
		y = self.model.forward(x_adv)
		J = self.loss(y,y_true)

		if x_adv.grad is not None:
			x_adv.grad.data.fill_(0)
		x_grad = torch.autograd.grad(J, x_adv)[0]

		if type(self.normalization) is int:
			x_norm = x_grad.view(x_grad.size()[0], -1)
			x_norm = F.normalize(x_norm,self.normalization)
			x_norm = x_norm.view(x_grad.size())
		elif self.normalization == "infty":
			x_norm = x_grad.sign_()
		else:
			n_norm = x_grad

		x_adv = x_adv + self.epsilon*x_norm

	def constraint(self, x_adv, x_original, p = 2):
		x_diff = x_original - x_adv
		x_norm = x_grad.view(x_grad.size()[0], -1)
		if type(p) is int:
			x_diff = x_diff / x_diff.norm(p, dim, True).clamp(min=eps).expand_as(x_diff)
		x_norm = x_norm.view(x_grad.size()) 


	def forward(self,x,y_true = None):
		x_adv = x.requires_grad()
		if y_true is None:
			y_true = self.sampleLabels(x)

		for step in range(step_size):
			self.step(x_adv,y_true)
			self.constraint(x_adv,x,p)

		x_adv = torch.clamp(x_adv, self.pixel_range[0], self.pixel_range[1])
		return x_adv

if __name__ == '__main__':
	from targets.mnistMLP import MNISTMLP
	from targets.datasets import MNIST
	model = torch.load("mnistMLP.pickle")
	dataset = MNIST()
	batch_size = 100

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


	dataiter = iter(dataset.testing(batch_size))
	images, labels = dataiter.next()
	images = images.cuda()
	

	gs = ItteratedGradientAttack(model,criterion,1,10,0.001)
	gs.visualize(images[:25],model,filename="gradient_0.1.png", diff_multiply = 10)
	successRate = gs.test(model,dataset.testing(batch_size))
	print("The success rate on a ResNet CIFAR with Gradient Method eps 0.1 is %.4f" % (successRate,))