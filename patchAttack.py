import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from tqdm import tqdm
from .attackTemplate import BaseAttack
from .ensembleUtils import ModelGrad


class AffineMaskSticker(BaseAttack):
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
		target,
		mask,
		targetShape,
		maxRotation,
		maxTranslation,
		scale,
		mean= 0.0,
		std = 0.1):
		super(AffineMaskSticker, self).__init__()
		self.my_target = target
		# If we wanted a png based mask, we can pass a filename rather than a numpy
		if isinstance(mask, str):
			from skimage import io, img_as_float

			mask = img_as_float(io.imread(mask))

			mask = mask.transpose(2,0,1)


		self.mask = torch.from_numpy(mask)
		self.mask = self.mask.type(torch.FloatTensor)
		self.mask = nn.Parameter(self.mask,requires_grad = False)

		# We want the gradient to be applied as if it were centered ao the mean, but for regularization to recenter it. So we let the thing be at 0
		self.mean = mean
		zero_mean = torch.zeros(self.mask.size())
		std = std*torch.ones(self.mask.size())
		self.sticker = nn.Parameter(torch.normal(mean,std))
		
		# Roughly in the middle
		x = (targetShape[-1] - self.mask.size()[-1])//2
		y = targetShape[-1] - (x + self.mask.size()[-1])
		z = (targetShape[-2] - self.mask.size()[-2])//2
		w = targetShape[-2] - (z + self.mask.size()[-2])
		self.pad = nn.ConstantPad3d((x,y,z,w,0,0), 0)

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

	def setBatchSize(self,batch_size):
		'''
		This preallocates the buffers for the random affine transformations
		'''
		self.batch_size = batch_size
		self.samples = torch.Tensor(batch_size,4).type(torch.FloatTensor)
		self.samples = nn.Parameter(self.samples,requires_grad=False)
		self.placedMask = 1-self.pad(self.mask)
		self.placedMask = self.placedMask.unsqueeze_(0)
		self.placedMask = self.placedMask.expand(self.batch_size,-1,-1,-1)
		self.placedMask = nn.Parameter(self.placedMask,requires_grad=False)
		self.aff = nn.Parameter(torch.zeros((self.batch_size,2,3)),requires_grad=False)

	@property
	def target(self):
		targetLabel = torch.full([self.batch_size],self.my_target,dtype=torch.long)
		targetLabel = targetLabel.to(self.aff.device)
		return targetLabel

	def __setAff__(self,S):
		'''
		Takes the random samples, S and converts it into 2d affine transformations
		'''
		self.aff[:, 0, 0] = (1/S[:,1])*S[:,0].cos()
		self.aff[:, 0, 1] = (1/S[:,1])*S[:,0].sin()
		self.aff[:, 1, 0] = (1/S[:,1])*(-S[:,0]).sin()
		self.aff[:, 1, 1] = (1/S[:,1])*S[:,0].cos()
		self.aff[:, 0, 2] = S[:,2]
		self.aff[:, 1, 2] = S[:,3]

	def forward(self,images):
		'''
		Places the sticker on the image with a random translation
		With a batch it applies the same translation.
		'''
		maskedSticker = self.mean + self.sticker 						# place it in the middle of the pixel space
		maskedSticker = torch.mul(self.sticker,self.mask)
		maskedSticker = self.pad(maskedSticker)
		maskedSticker = maskedSticker.unsqueeze_(0)
		maskedSticker = maskedSticker.expand(self.batch_size,-1,-1,-1)

		self.samples = self.samples.random_(0,1000)
		S = F.linear(self.samples, self.boundaries,self.offsets)
		self.__setAff__(S)


		affineGrid = F.affine_grid(self.aff,images.size())
		placedMask = F.grid_sample(self.placedMask,affineGrid,padding_mode='border')
		maskedSticker = F.grid_sample(maskedSticker,affineGrid)

		maskedImage = torch.mul(images,placedMask)
		stickered = maskedSticker + maskedImage

		return stickered

	@property
	def usesLabels(self):
		return False

def trainPatch_cuda(masker,models,loader,optimizer,criterion,epochs,update_rate=20):
	epoch_size = len(loader)
	targetLabel = masker.target



	for epoch in range(epochs):
		epoch_loss = torch.zeros((1,))
		epoch_loss = epoch_loss.cuda()
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
				epoch_loss += update_loss
				epoch_loss, update_loss = epoch_loss.cpu(), update_loss.cpu()
				dataIterator.set_description(
					"update loss: %.3f, epoch loss: %.3f" % (
						update_loss[0] / update_rate,
						epoch_loss[0]/(i + 1),
						))
				update_loss.zero_()
				epoch_loss, update_loss = epoch_loss.cuda(), update_loss.cuda()

			
		epoch_loss = epoch_loss.cpu()
		print("Epoch %d/%d loss: %.4f" % (epoch+1,epochs,epoch_loss[0]/epoch_size))
				
		#if (epoch == 0 or epoch == 9):
			#imshow(stickered.clone().detach())


def trainPatch(masker,models,loader,optimizer,criterion,epochs,update_rate=20):
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


class StickerTrainer():
	def __init__(self,masker,models,losses,maskerLocation = 0):
		self.masker = masker
		self.masker.cuda(maskerLocation)
		self.maskerLocation = maskerLocation
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
		targetLabels = torch.cuda.comm.broadcast(targetLabel,[self.maskerLocation,1])

		for epoch in range(epochs):
			epoch_loss = torch.zeros((1,))
			update_loss = torch.zeros((1,))

			dataIterator = tqdm(enumerate(dataLoader, 0),total = epoch_size)
			dataIterator.set_description("update loss: %.3f, epoch loss: %.3f" % (0,0))
			for i, data in dataIterator:
				images, labels = data
				images = images.cuda(self.maskerLocation)

				stickered = self.masker(images)
				stickers = torch.cuda.comm.broadcast(stickered,[self.maskerLocation,1])

				stickerGrads = nn.parallel.parallel_apply(self.wrappedModels, zip(stickers,targetLabels))
				
				stickerGrad = torch.cuda.comm.reduce_add(stickerGrads, 
					destination=self.maskerLocation)

				optimizer.zero_grad()
				stickered.backward(stickerGrad)
				optimizer.step()
				# print statistics
				'''
				update_loss += loss
				if i % update_rate == update_rate - 1:    # print every 500 mini-batches
					epoch_loss += update_loss
					dataIterator.set_description(
						"update loss: %.3f, epoch loss: %.3f" % (
							update_loss[0] / update_rate,
							epoch_loss[0]/(i + 1),
							))
					update_loss.zero_()
				'''
				
			print("Epoch %d/%d loss: %.4f" % (epoch+1,epochs,epoch_loss[0]/epoch_size))
			
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