import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from tqdm import tqdm
from ..attackTemplate import BaseAttack
from ..utils import *
from ..ensembleUtils import *
from __init__ import *
import itertools

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
		self.device = torch.device("cpu")
		self.batchSize = 0
		self.targetLabel = None

	def cuda(self,i):
		self.device = torch.device("cuda:%d"%(i,))
		self.model = self.model.to(self.device)
		return self

	def device(self,dev):
		self.device = torch.device(dev)
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

	def __call__(self,images):
		if images.device != self.device:
			self.device = images.device
		if self.batchSize != images.size()[0]:
			self.__setBatchSize__(images.size()[0])
		y = self.model(images)
		return self.lossfn(y,self.targetLabel)

def trainStep(sticker,placer,models,images):
	sticker_adv = sticker.requires_grad_()
	stickered = placer(images,sticker)
	loss = models[0](stickered)
	for model in models[1:]:
		loss += model(stickered)
	grad = torch.autograd.grad(loss, sticker_adv)[0]
	del loss
	return grad

def testStep(sticker,placer,model,images,targetLabel):
	sticker = sticker.requires_grad_()
	stickered = placer(images,sticker)
	y = model(stickered)
	val, pred = torch.max(y.data, 1)
	correct = ((pred == targetLabel).sum().item())
	total = images.size(0)
	return total,correct

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
		self.target = target
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

	def train(self,dataLoader,optimizer,epochs, num_steps = None,update_rate = 50,targetModel = None,root = "Adam/"):
		# Setup threads

		epoch_size = len(dataLoader)
		images,labels = iter(dataLoader).next()

		if targetModel is not None:
			targetModel = targetModelWrap(targetModel,self.target)

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
				
				sticker.backward((1.0/len(self.models))*grad)
				optimizer.step()
				self.sticker.clamp()

				if i % (update_rate-1) == 0 and targetModel is not None:
					total, correct = 0.0,0.0
					testloader = itertools.islice(dataLoader, 50)
					for testData in testloader:
						testImage, labels = testData
						t, c = targetModel(testImage)
						total += t
						correct += c
					sticker = self.sticker()
					print(sticker.max(),sticker.min(),sticker.var(),sticker.mean())
					print(testImage.max(),testImage.min(),testImage.var(),testImage.mean())
					trainedError = correct/total

					self.sticker.save(root+"ai_sticker_iter_%d_%.3f_res101_50.png"%(i+1,trainedError))
					torch.save(self.sticker,root+"sticker_iter_%d_%.3f_res101_50.pkl"%(i+1,trainedError))