import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from tqdm import tqdm

from attackTemplate import BaseAttack

class RawGradientAttack(BaseAttack):
	def __init__(self,model, loss, epsilon):
		super(RawGradientAttack, self).__init__()
		self.model = model
		self.loss = loss
		self.epsilon = epsilon

	def forward(self,x,y_true):
		x_adv = x.requires_grad_()

		self.model.zero_grad()
		y = self.model.forward(x_adv)
		J = self.loss(y,y_true)

		if x_adv.grad is not None:
			x_adv.grad.data.fill_(0)

		x_grad = torch.autograd.grad(J, x_adv, allow_unused=True)[0]
		print torch.mean(torch.norm(x_grad,2,0, keepdim=True))

		return x + self.epsilon*x_grad

	

class IteratedRawGradientAttack(BaseAttack):
	def __init__(self,model, loss, epsilon, n):
		super(IteratedRawGradientAttack, self).__init__()
		self.model = model
		self.loss = loss
		self.epsilon = epsilon
		self.n = n

	def forward(self,x,y_true):
		x_adv = x.requires_grad_()

		for i in range(self.n):
			y = self.model.forward(x_adv)
			J = self.loss(y,y_true)

			if x_adv.grad is not None:
				x_adv.grad.data.fill_(0)

			x_grad = torch.autograd.grad(J, x_adv, allow_unused=True)[0]
			x_adv = x_adv + self.epsilon*x_grad

		return x_adv

	@property
	def usesLabels(self):
		return True

class GradientAttack(BaseAttack):
	def __init__(self,model, loss, epsilon):
		super(GradientAttack, self).__init__()
		self.model = model
		self.loss = loss
		self.epsilon = epsilon

	def forward(self,x,y_true):
		x_adv = x.requires_grad_()


		y = self.model.forward(x_adv)
		J = self.loss(y,y_true)

		if x_adv.grad is not None:
			x_adv.grad.data.fill_(0)

		x_grad = torch.autograd.grad(J, x_adv, allow_unused=True)[0]
		x_grad = x_grad/torch.norm(x_grad,2,0, keepdim=True)

		return x + self.epsilon*x_grad

	@property
	def usesLabels(self):
		return True

class GradientSignAttack(BaseAttack):
	def __init__(self,model, loss, epsilon):
		super(GradientSignAttack, self).__init__()
		self.model = model
		self.loss = loss
		self.epsilon = epsilon

	def forward(self,x,y_true):
		x_adv = x.requires_grad_()


		y = self.model.forward(x_adv)
		J = self.loss(y,y_true)

		if x_adv.grad is not None:
			x_adv.grad.data.fill_(0)

		x_grad = torch.autograd.grad(J, x_adv)[0]

		return x + self.epsilon*x_grad.sign_()

	@property
	def usesLabels(self):
		return True

if __name__ == '__main__':
	from utils import *
	import torch.optim as optim
	from skimage import io, transform, img_as_float
	from skimage.color import gray2rgb

	use_cifar = False

	if use_cifar:
		from cifar10 import CIFAR10ResNet,residual
		from datasets import CIFAR10
		model = torch.load("cifarnetbn.pickle")
		dataset = CIFAR10()
		batch_size = 400
	else:
		from mnist import MNISTNet
		from datasets import MNIST
		model = torch.load("mnist.pickle")
		dataset = MNIST()
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

	gs = IteratedRawGradientAttack(model,criterion,1,1)

	print testNonTargetedAttack(model,dataset.testing(batch_size),gs)