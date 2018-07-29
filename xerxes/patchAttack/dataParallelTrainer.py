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

def trainStep(sticker,placer,model,lossfn,images,targetLabel):
	sticker_adv = sticker.requires_grad_()
	stickered = placer(images,sticker)
	loss = lossfn(model(stickered),targetLabel)
	grad = torch.autograd.grad(loss, sticker_adv)[0]
	return grad,loss

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
		self.models = models
		self.losses = losses

		self.batchSize = 0
		self.targetLabel = None

		self.numDev = torch.cuda.device_count()
		if self.numDev > 0:
			self.cuda = True
			# Replicate the placer
			self.placers = nn.parallel.replicate(placer.cuda(), list(range(self.numDev)))
			# Move all the models to GPUs. We're an ensemble model parallel 
			self.cudaModels = []
			self.cudaLosses = []
			self.trainingSteps = [trainStep for i in range(self.numDev)]
			self.testingSteps = [testStep for i in range(self.numDev)]
			for model,loss in zip(self.models,self.losses):
				self.cudaModels.append(nn.parallel.replicate(model.cuda(), list(range(self.numDev))))
				self.cudaLosses.append(nn.parallel.replicate(loss.cuda(), list(range(self.numDev))))

			self.sticker = self.sticker.cuda()
		else:
			self.cuda = False

	def __setBatchSize__(self,batchSize):
		self.batchSize = batchSize
		if self.targetLabel is not None:
			del self.targetLabel
		self.targetLabel = torch.full([self.batchSize],
				self.target,
				dtype=torch.long)
		if self.cuda:
			self.targetLabel = self.targetLabel.cuda()
			self.targetLabel = torch.cuda.comm.scatter(self.targetLabel,list(range(self.numDev)))


	def train(self,dataLoader,optimizer,epochs, num_steps = None,update_rate = 50,targetModel = None,root = None):
		# Setup threads

		epoch_size = len(dataLoader)
		images,labels = iter(dataLoader).next()
		if images.size()[0] != self.batchSize:
			self.__setBatchSize__(images.size()[0])


		if targetModel is not None and self.cuda:
			targetModel = nn.parallel.replicate(targetModel.cuda(), list(range(self.numDev)))

		for epoch in range(epochs):
			epoch_loss = torch.zeros((1,))

			dataIterator = tqdm(enumerate(dataLoader, 0),total = epoch_size)
			dataIterator.set_description("current validation: %.3f" % (0,))
			for i, data in dataIterator:
				if num_steps is not None and i > num_steps: 
					break
				images, labels = data

				if self.cuda:
					images = images.cuda()
					images = torch.cuda.comm.scatter(images,list(range(self.numDev)))
					for model,loss in zip(self.cudaModels,self.cudaLosses):
						sticker = self.sticker()
						stickers = torch.cuda.comm.broadcast(sticker,list(range(self.numDev)))
						gradsAndLosses = nn.parallel.parallel_apply(
							self.trainingSteps,zip(
								stickers,
								self.placers,
								model,
								loss,
								images,
								self.targetLabel))

						grads,losses = zip(*gradsAndLosses)
						grad = torch.cuda.comm.reduce_add(grads,0)
						
						optimizer.zero_grad()
						sticker.backward(grad)
						optimizer.step()
						self.sticker.clamp()
				
				#print(images[0].size(),images[1].size())
				

				if i % (update_rate-1) == 0 and targetModel is not None:
					total, correct = 0.0,0.0
					testloader = itertools.islice(dataLoader, 50)
					if self.cuda:
						for testData in testloader:
							testImage, labels = testData
							stepTotals = nn.parallel.parallel_apply(
								self.testingSteps,zip(
									stickers,
									self.placers,
									model,
									images,
									self.targetLabel))
							for t,c in stepTotals:
								total += t
								correct += c
					else:
						for testData in testloader:
							testImage, labels = testData
							t, c = targetModel(testImage)
							total += t
							correct += c
					sticker = self.sticker()
					print(sticker.max(),sticker.min(),sticker.var(),sticker.mean())
					print(testImage.max(),testImage.min(),testImage.var(),testImage.mean())
					trainedError = correct/total
					dataIterator.set_description("current validation: %.3f" % (trainedError,))
					if root is not None:
						self.sticker.save(root+"ai_sticker_iter_%d_%.3f_res101_50.png"%(i+1,trainedError))
						torch.save(self.sticker,root+"sticker_iter_%d_%.3f_res101_50.pkl"%(i+1,trainedError))