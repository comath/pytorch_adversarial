import torch.nn as nn
import torch

class ModelGrad(nn.Module):
	def __init__(self,model,loss):
		super(ModelGrad, self).__init__()
		self.model = model
		self.loss = loss

	def setX(self,x):
		self.x = x

	def setY(self,y):
		self.y = y

	def forward(self,x,y_true = None):
		x_adv = x.requires_grad_()
		y = self.model(x_adv)
		if y_true is None:
			loss = self.loss(y, self.y)
		else:
			loss = self.loss(y, y_true)
		#return torch.autograd.grad(loss, x_adv)[0]
		return loss