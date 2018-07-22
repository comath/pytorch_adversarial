import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from tqdm import tqdm
from ..attackTemplate import BaseAttack
from ..utils import *
from __init__ import *

class PatchModelWrap():
	'''
	Wraps a model so that we can easily hand it a patch and get back the gradient
	'''
	def __init__(self,model,lossfn,target):
		#super(PatchModelWrap, self).__init__()
		self.model = model
		#self.add_module("model",model)
		self.lossfn = lossfn
		self.target = target
		self.updateLoss = torch.zeros(1)
		self.device = torch.device("cpu")
		self.batchSize = 0
		self.targetLabel = None

	def cuda(self,i):
		self.device = torch.device("cuda:%d"%(i,))
		self.updateLoss = self.updateLoss.to(self.device)
		self.model = self.model.to(self.device)
		return self

	def device(self,dev):
		self.device = torch.device(dev)
		self.updateLoss = self.updateLoss.to(self.device)
		self.model = self.model.to(self.device)
		return self

	def __setBatchSize__(self,batchSize):
		self.batchSize = batchSize
		if self.targetLabel is not None:
			del self.targetLabel
		self.targetLabel = torch.full([self.batchSize],
				self.target,
				dtype=torch.long,
				device=self.device)

	def getRunningLoss(self):
		retVal = self.updateLoss
		self.updateLoss[0] = 0
		return retVal

	def __call__(self,images):
		if images.device != self.device:
			self.device = images.device
		if self.batchSize != images.size()[0]:
			self.__setBatchSize__(images.size()[0])
		y = self.model(images)
		loss = self.lossfn(y,self.targetLabel)
		self.updateLoss = self.updateLoss + loss
		return loss

def trainStep(sticker,placer,models,images):
	sticker_adv = sticker.requires_grad_()
	stickered = placer(images,sticker)
	loss = models[0](stickered)
	for model in models[1:]:
		loss += model(stickered)
	grad = torch.autograd.grad(loss, sticker_adv)[0]
	return grad

class StickerTrainer():
	'''
	Helper class to train up an adversarial sticker. 
	'''
	def __init__(self,
			sticker,
			models,
			losses,
			target,
			placer):

		self.sticker = sticker
		self.placer = placer
		self.models = [PatchModelWrap(model,loss,target) for model,loss in zip(models,losses)]
		self.cuda = False

		self.numDev = torch.cuda.device_count()
		if self.numDev > 0:
			self.cuda = True
			# Replicate the placer
			self.placers = nn.parallel.replicate(placer.cuda(), list(range(self.numDev)))
			# Move all the models to GPUs. We're an ensemble model parallel 
			self.cudaModels = [[] for i in range(self.numDev)]
			self.trainingSteps = [trainStep for i in range(self.numDev)]
			for i in range(len(self.models)):
				self.cudaModels[i%self.numDev].append(self.models[i].cuda(i%self.numDev))

			self.sticker = self.sticker.cuda()
		else:
			self.cuda = False

	def train(self,dataLoader,optimizer,epochs, num_steps = None,update_rate = 20):
		# Setup threads

		epoch_size = len(dataLoader)
		images,labels = iter(dataLoader).next()

		for epoch in range(epochs):
			epoch_loss = torch.zeros((1,))

			dataIterator = tqdm(enumerate(dataLoader, 0),total = epoch_size)
			dataIterator.set_description("update loss: %.3f, epoch loss: %.3f" % (0,0))
			for i, data in dataIterator:
				if num_steps is not None and i > num_steps: 
					break
				sticker = self.sticker()
				images, labels = data

				if self.cuda:
					images = images.cuda()
					images = torch.cuda.comm.scatter(images,list(range(self.numDev)))
					stickers = torch.cuda.comm.broadcast(sticker,list(range(self.numDev)))
					
					grads = nn.parallel.parallel_apply(
							self.trainingSteps,zip(stickers,
							self.placers,
							self.cudaModels,
							images))
					grad = torch.cuda.comm.reduce_add(grads,0)
				else:
					pass
				#print(images[0].size(),images[1].size())
				optimizer.zero_grad()
				
				sticker.backward(grad)
				optimizer.step()
				self.sticker.clamp()
