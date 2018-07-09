import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from tqdm import tqdm
from .attackTemplate import BaseAttack
from .ensembleUtils import ModelGrad
from .utils import *

class AdversarialSticker(nn.Module):
	def __init__(self,mask,mean,std = 0.1):
		super(AdversarialSticker, self).__init__()

		if isinstance(mask, str):
			from skimage import io, img_as_float
			mask = img_as_float(io.imread(mask))
			mask = mask.transpose(2,0,1)

		self.mask = torch.from_numpy(mask)
		self.mask = self.mask.type(torch.FloatTensor)
		self.mask = nn.Parameter(self.mask,requires_grad = False)

		self.mean = mean
		zero_mean = torch.zeros(self.mask.size())
		std = std*torch.ones(self.mask.size())
		self.sticker = nn.Parameter(torch.normal(mean,std))

	def forward(self):
		maskedSticker = self.mean + self.sticker 						# place it in the middle of the pixel space
		maskedSticker = torch.mul(self.sticker,self.mask)
		return maskedSticker

	def size(self):
		return self.mask.size()

class AffinePlacer(nn.Module):
	"""
	Given a mask that can be a png image or an numpy array this creates a sticker. After creation
	you must set the mini-batch size, before shipping it to the GPU.

	This uses the affine grid generator and grid sampler to generate the transformations	


	Args:
		target: We assume a 1-hot encoding with a target label, give the index of that label
        mask: either a numpy array or a the filename of a png image, of shape `(C,H_{mask},W_{mask})`
		targetShape: the shape that this mask needs to fit into
        maxRotation: The mask will be rotated within (-maxRotation,maxRotation) degrees
        maxTranslation: The mask will be translated (-maxTranslation,maxTranslation) 
        	within the [-1,1]x[-1,1] image space
        scale: If this is a float then the mask will be scaled by a number randomly sampled from
        	(1-scale,1+scale), if it's a pair (x,y) then it will sampled from that interval

    Shape:
        - Input: `(N,C,H,W)`
        - Output: `(N,C,H,W)`  (same shape as input)

    Attributes:
        sticker: the learnable weights of the sticker of shape
            `(C,H_{mask},W_{mask})` (same as mask)
        mask:   not-learnable mask of shape `(C,H_{mask},W_{mask})`
        
    Examples::
		
	"""
	def __init__(self,
		mask,
		targetShape,
		maxRotation,
		maxTranslation,
		scale):
		super(AffinePlacer, self).__init__()
		# If we wanted a png based mask, we can pass a filename rather than a numpy
		# We want the gradient to be applied as if it were centered ao the mean, but for regularization to recenter it. So we let the thing be at 0
		
		if isinstance(mask, str):
			from skimage import io, img_as_float
			mask = img_as_float(io.imread(mask))
			mask = mask.transpose(2,0,1)

		self.mask = torch.from_numpy(mask)
		self.mask = self.mask.type(torch.FloatTensor)
		self.mask = nn.Parameter(self.mask,requires_grad = False)
		
		# Roughly in the middle
		x = (targetShape[-1] - self.mask.size()[-1])//2
		y = targetShape[-1] - (x + self.mask.size()[-1])
		z = (targetShape[-2] - self.mask.size()[-2])//2
		w = targetShape[-2] - (z + self.mask.size()[-2])
		self.pad = nn.ConstantPad3d((x,y,z,w,0,0), 0)
		self.add_module("pad",self.pad)

		# The boundaries of the rotation, translation and scaling
		self.boundaries = nn.Parameter(torch.zeros(4,4),requires_grad=False)
		self.offsets = nn.Parameter(torch.zeros(4),requires_grad=False)
		self.boundaries[0,0] = (1.0/1000.0)*math.pi*maxRotation/90
		self.offsets[0] = -math.pi*maxRotation/180
		if scale is float:
			self.boundaries[1,1] = (1.0/1000.0)*2*scale
			self.offsets[1] = 1-scale
		else:
			self.boundaries[1,1] = (1.0/1000.0)*(scale[1] - scale[0])
			self.offsets[1] = scale[0]
		self.boundaries[2,2] = (1.0/1000.0)*2*maxTranslation		#x
		self.offsets[2] = -maxTranslation
		self.boundaries[3,3] = (1.0/1000.0)*2*maxTranslation		#y
		self.offsets[3] = -maxTranslation

		self.batch_size = 0
		self.placedMask = None
		self.aff = None
		self.samples = None

	def __setBatchSize__(self,batch_size):
		'''
		This preallocates the buffers for the random affine transformations
		'''
		self.batch_size = batch_size
		if self.samples is not None:
			del self.samples
		self.samples = torch.Tensor(batch_size,4).type(torch.FloatTensor)
		self.samples = self.samples.to(self.boundaries.device)
		self.samples = nn.Parameter(self.samples,requires_grad=False)
		if self.placedMask is not None:
			del self.placedMask
		self.placedMask = 1-self.pad(self.mask)
		self.placedMask = self.placedMask.unsqueeze_(0)
		self.placedMask = self.placedMask.expand(self.batch_size,-1,-1,-1)
		self.placedMask = self.placedMask.to(self.boundaries.device)
		self.placedMask = nn.Parameter(self.placedMask,requires_grad=False)
		if self.aff is not None:
			del self.aff
		self.aff = torch.zeros((self.batch_size,2,3))
		self.aff = self.aff.to(self.boundaries.device)
		self.aff = nn.Parameter(self.aff,requires_grad=False)

	def __setAff__(self,S,scale = None):
		'''
		Takes the random samples, S and converts it into 2d affine transformations
		'''
		if scale is None:
			self.aff[:, 0, 0] = (1/S[:,1])*S[:,0].cos()
			self.aff[:, 0, 1] = (1/S[:,1])*S[:,0].sin()
			self.aff[:, 1, 0] = (1/S[:,1])*(-S[:,0]).sin()
			self.aff[:, 1, 1] = (1/S[:,1])*S[:,0].cos()
			self.aff[:, 0, 2] = S[:,2]
			self.aff[:, 1, 2] = S[:,3]
		else:
			self.aff[:, 0, 0] = (1/scale)*S[:,0].cos()
			self.aff[:, 0, 1] = (1/scale)*S[:,0].sin()
			self.aff[:, 1, 0] = (1/scale)*(-S[:,0]).sin()
			self.aff[:, 1, 1] = (1/scale)*S[:,0].cos()
			self.aff[:, 0, 2] = S[:,2]
			self.aff[:, 1, 2] = S[:,3]

	def forward(self,images,sticker,scale = None):
		'''
		Places the sticker on the image with a random translation
		With a batch it applies the same translation.
		'''
		if self.batch_size != images.size()[0]:
			self.__setBatchSize__(images.size()[0])

		
		maskedSticker = self.pad(sticker)
		maskedSticker = maskedSticker.unsqueeze_(0)
		maskedSticker = maskedSticker.expand(self.batch_size,-1,-1,-1)

		self.samples = self.samples.random_(0,1000)
		S = F.linear(self.samples, self.boundaries,self.offsets)
		self.__setAff__(S)


		affineGrid = F.affine_grid(self.aff,images.size())
		placedMask = 1-F.grid_sample(1-self.placedMask,affineGrid,padding_mode='zeros')
		maskedSticker = F.grid_sample(maskedSticker,affineGrid)

		maskedImage = torch.mul(images,placedMask)
		stickered = maskedSticker + maskedImage		

		return stickered

class StickerAttack(BaseAttack):
	def __init__(self,sticker,placer,target):
		super(StickerAttack, self).__init__()

		self.sticker = sticker
		self.placer = placer
		self.my_target = target
		self.batchSize = 0
		self.targetLabel = None

	def __setBatchSize__(self,batchSize):
		self.batchSize = batchSize
		if self.targetLabel is not None:
			del self.targetLabel
		self.targetLabel = torch.full([self.batchSize],self.my_target,dtype=torch.long)
		self.targetLabel = self.targetLabel.to(self.sticker.sticker.device)

	def forward(self,images,scale = None):
		if self.batchSize != images.size()[0]:
			self.__setBatchSize__(images.size()[0])
		sticker = self.sticker()
		placedSticker = self.placer(images,sticker,scale)
		return placedSticker

	@property
	def target(self):
		if self.targetLabel is None:
			raise AttributeError("set the batch size or call forward first")
		return self.targetLabel

	@property
	def usesLabels(self):
		return False


def trainPatch_cuda(masker,models,loader,optimizer,criterion,epochs,update_rate=20):
	epoch_size = len(loader)
	targetLabel = masker.target



	for epoch in range(epochs):
		epoch_loss = torch.zeros((1,))
		update_loss = torch.zeros((1,))
		update_loss = update_loss.cuda()

		dataIterator = tqdm(enumerate(loader, 0),total = epoch_size)
		dataIterator.set_description("update loss: %.3f, epoch loss: %.3f" % (0,0))
		for i, data in dataIterator:
			images, labels = data
			images = images.cuda(async=True)

			optimizer.zero_grad()
			stickered = masker(images)
			stickerGrad = torch.zeros(stickered.size())
			for model in models:
				output = model(stickered)
				loss = criterion(output, targetLabel)
				stickerGrad += torch.autograd.grad(loss, stickered)[0]

			stickered.backward(stickerGrad)
			optimizer.step()
			# print statistics
			update_loss += loss
			if i % update_rate == update_rate - 1:    # print every 500 mini-batches
				update_loss = update_loss.cpu()
				epoch_loss += update_loss
				dataIterator.set_description(
					"update loss: %.3f, epoch loss: %.3f" % (
						update_loss[0] / update_rate,
						epoch_loss[0]/(i + 1),
						))
				update_loss.zero_()
				update_loss = update_loss.cuda()

			
		epoch_loss = epoch_loss.cpu()
		print("Epoch %d/%d loss: %.4f" % (epoch+1,epochs,epoch_loss[0]/epoch_size))
				
		#if (epoch == 0 or epoch == 9):
			#imshow(stickered.clone().detach())


def trainPatch(masker,model,loader,optimizer,criterion,epochs,update_rate=20):
	epoch_size = len(loader)
	targetLabel = masker.target
	for epoch in range(epochs):
		epoch_loss = torch.zeros((1,))
		update_loss = torch.zeros((1,))

		dataIterator = tqdm(enumerate(loader, 0),total = epoch_size)
		dataIterator.set_description("update loss: %.3f, epoch loss: %.3f" % (0,0))
		for i, data in dataIterator:
			images, labels = data

			stickered = masker(images)

			stickerGrad = torch.zeros(stickered.size())
			for model in models:
				output = model(stickered)
				loss = criterion(output, targetLabel)
				stickerGrad += torch.autograd.grad(loss, stickered)[0]

			optimizer.zero_grad()
			stickered.backward(stickerGrad)
			optimizer.step()
			# print statistics
			update_loss += loss
			if i % update_rate == update_rate - 1:    # print every 500 mini-batches
				epoch_loss += update_loss
				dataIterator.set_description(
					"update loss: %.3f, epoch loss: %.3f" % (
						update_loss[0] / update_rate,
						epoch_loss[0]/(i + 1),
						))
				update_loss.zero_()

			
		print("Epoch %d/%d loss: %.4f" % (epoch+1,epochs,epoch_loss[0]/epoch_size))
				
		#if (epoch == 0 or epoch == 9):
			#imshow(stickered.clone().detach())


def trainingStep(sticker,placer,model,lossfn,images,targetLabel):
	sticker = sticker.requires_grad_()
	stickered = placer(images,sticker)
	y = model(stickered)
	loss = lossfn(y,targetLabel)
	stickerGrad = torch.autograd.grad(loss, sticker)[0]
	return stickerGrad,loss

def averageStickers(maskers):
	stickers = [masker.sticker for masker in maskers]
	stickers = torch.cuda.comm.reduce_add(stickers,0)
	sticker  = stickers[0]/len(stickers)
	stickers = torch.cuda.comm.broadcast(stickers,[0,1])
	for masker,sticker in zip(maskers,stickers):
		masker.sticker.data = sticker.data 

class StickerTrainer():
	def __init__(self,stickerAttack,models,losses):
		self.attack = stickerAttack
		self.sticker = stickerAttack.sticker
		self.sticker.cuda(0)
		self.placers = nn.parallel.replicate(stickerAttack.placer, [0,1])
		self.trainingSteps = [trainingStep for i in range(2)]
		self.lossFns = losses
		self.models = models
		assert len(models) == torch.cuda.device_count()
		assert len(models) == len(losses)
		

	def train(self,dataLoader,optimizer,epochs,update_rate = 20):
		# Setup threads

		epoch_size = len(dataLoader)
		images,labels = iter(dataLoader).next()
		self.attack.__setBatchSize__(images.size()[0])
		targetLabels = self.attack.target
		targetLabels = torch.cuda.comm.scatter(targetLabels,[0,1])

		for epoch in range(epochs):
			epoch_loss = torch.zeros((1,))
			update_loss = torch.zeros((1,))
			update_loss = update_loss.cuda()

			dataIterator = tqdm(enumerate(dataLoader, 0),total = epoch_size)
			dataIterator.set_description("update loss: %.3f, epoch loss: %.3f" % (0,0))
			for i, data in dataIterator:
				images, labels = data
				images = torch.cuda.comm.scatter(images,[0,1])

				#print(images[0].size(),images[1].size())
				sticker = self.sticker()
				stickers = torch.cuda.comm.broadcast(sticker,[0,1])
				optimizer.zero_grad()
				gradsAndLoses = nn.parallel.parallel_apply(self.trainingSteps,zip(stickers,self.placers,self.models,self.lossFns,images,targetLabels))
				grads, losses = zip(*gradsAndLoses)
				grad = torch.cuda.comm.reduce_add(grads,0)
				sticker.backward(grad)
				optimizer.step()
				#averageStickers(self.maskers)
				
					
			#if (epoch == 0 or epoch == 9):
				#imshow(stickered.clone().detach())
'''
def patchTrainingStep(bits,images):
	masker,wrappedModel = bits
	stickered = self.masker(images)
	stickers = torch.cuda.comm.scatter(stickered,[0,1])

	stickerGrads = wrappedModel(stickers)

	stickered.backward(stickerGrad)
	optimizer.step()

class StickerTrainer2():
	def __init__(self,masker,models,losses,batch_size):
		self.maskers = replicate(masker,[0,1])
		self.wrappedModels = []
		assert len(models) == torch.cuda.device_count()
		assert len(models) == len(losses)
		for i in range(torch.cuda.device_count()):
			model = models[i]
			model.cuda(i)
			wrappedModel = ModelGrad(model,losses[i])
			self.wrappedModels.append(wrappedModel)

	def train(self,dataLoader,optimizer,epochs):
		# Setup threads

		epoch_size = len(dataLoader)
		targetLabel = self.masker.target.cuda(self.maskerLocation)
		targetLabels = torch.cuda.comm.scatter(targetLabel,[0,1])

		bits = zip(self.maskers,self.wrappedModels)

		for (masker,model), labels in zip(bits,targetLabels):
			masker.setBatchSize(len(labels))
			model.setY(labels)

		for epoch in range(epochs):
			epoch_loss = torch.zeros((1,))
			update_loss = torch.zeros((1,))

			dataIterator = tqdm(enumerate(dataLoader, 0),total = epoch_size)
			dataIterator.set_description("update loss: %.3f, epoch loss: %.3f" % (0,0))
			for i, data in dataIterator:
				optimizer.zero_grad()
				images, labels = data
				images = torch.cuda.comm.scatter(images, [0,1])

				stickerGrads = nn.parallel.parallel_apply(patchTrainingStep, zip(bits,targetLabels))
				
				# print statistics
				
				update_loss += loss
				if i % update_rate == update_rate - 1:    # print every 500 mini-batches
					epoch_loss += update_loss
					dataIterator.set_description(
						"update loss: %.3f, epoch loss: %.3f" % (
							update_loss[0] / update_rate,
							epoch_loss[0]/(i + 1),
							))
					update_loss.zero_()
				
				
			print("Epoch %d/%d loss: %.4f" % (epoch+1,epochs,epoch_loss[0]/epoch_size))
			
			#if (epoch == 0 or epoch == 9):
				#imshow(stickered.clone().detach())

'''