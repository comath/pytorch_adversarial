import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import os
import random
import torchvision 
from imgNetDict import lookup
import torch.optim as optim
import math

from skimage import io, transform, img_as_float
from skimage.color import gray2rgb

from mnist import MNISTNet
from datasets import MNIST
from utils import *


class AffineMaskSticker(nn.Module):
	"""
	Given a mask that can be a png image or an numpy array this creates a sticker. After creation
	you must set the mini-batch size, before shipping it to the GPU.	


	Args:
        mask: either a numpy array or a the filename of a png image, of shape `(C,H_{mask},W_{mask})`
		targetShape: the shape that this mask needs to fit into
        maxRotation: The mask will be rotated within (-maxRotation,maxRotation) degrees
        maxTranslation: The mask will be translated (-maxTranslation,maxTranslation) 
        	within the [-1,1]x[-1,1] image space
        maxScale: The mask will be scaled by (1-maxScale,1+maxScale) 

    Shape:
        - Input: `(N,C,H,W)`
        - Output: `(N,C,H,W)`  (same shape as input)

    Attributes:
        sticker: the learnable weights of the sticker of shape
            `(C,H_{mask},W_{mask})` (same as mask)
        mask:   not-learnable mask of shape `(C,H_{mask},W_{mask})`
        
    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
	"""
	def __init__(self, 
		mask,
		targetShape,
		maxRotation,
		maxTranslation,
		maxScale,
		mean= 0.5,
		std = 0.05):
		super(AffineMaskSticker, self).__init__()
		# If we wanted a png based mask, we can pass a filename rather than a 
		if isinstance(mask, basestring):
			mask = img_as_float(io.imread(mask))

		self.mask = torch.from_numpy(mask)
		self.mask = self.mask.type(torch.FloatTensor)
		self.mask = nn.Parameter(self.mask,requires_grad = False)

		mean = mean*torch.ones(self.mask.size())
		std = std*torch.ones(self.mask.size())
		self.sticker = nn.Parameter(torch.normal(mean,std))
		
		# Roughly in the middle
		x = (targetShape[-2] - self.mask.size()[-2])/2
		y = targetShape[-2] - (x + self.mask.size()[-2])
		z = (targetShape[-1] - self.mask.size()[-1])/2
		w = targetShape[-1] - (z + self.mask.size()[-1])
		self.pad = nn.ConstantPad3d((x,y,z,w,0,0), 0)

		# The boundaries of the rotation, translation and scaling
		self.boundaries = nn.Parameter(torch.zeros(4,4),requires_grad=False)
		self.boundaries[0,0] = (1.0/1000.0)*math.pi*maxRotation/90
		self.boundaries[1,1] = (1.0/1000.0)*2*maxScale
		self.boundaries[2,2] = (1.0/1000.0)*2*maxTranslation		#x
		self.boundaries[3,3] = (1.0/1000.0)*2*maxTranslation		#y

		self.offsets = nn.Parameter(torch.zeros(4),requires_grad=False)
		self.offsets[0] = -math.pi*maxRotation/180
		self.offsets[1] = 1
		self.offsets[2] = -maxTranslation			#x
		self.offsets[3] = -maxTranslation			#y

	def setBatchSize(self,batch_size):
		self.batch_size = batch_size
		self.samples = torch.Tensor(batch_size,4).type(torch.FloatTensor)
		self.samples = nn.Parameter(self.samples,requires_grad=False)
		self.placedMask = 1-self.pad(self.mask)
		self.placedMask = self.placedMask.unsqueeze_(0)
		self.placedMask = self.placedMask.expand(self.batch_size,-1,-1,-1)
		self.placedMask = nn.Parameter(self.placedMask,requires_grad=False)
		self.aff = nn.Parameter(torch.zeros((self.batch_size,2,3)),requires_grad=False)

	def forward(self,images):
		'''
		Places the sticker on the image with a random translation
		With a batch it applies the same translation.
		'''
		maskedSticker = torch.mul(self.sticker,self.mask)
		maskedSticker = self.pad(maskedSticker)
		maskedSticker = maskedSticker.unsqueeze_(0)
		maskedSticker = maskedSticker.expand(self.batch_size,-1,-1,-1)

		self.samples = self.samples.random_(0,1000)
		S = F.linear(self.samples, self.boundaries,self.offsets)
		# Rotations
		self.aff[:, 0, 0] = S[:,1]*S[:,0].cos()
		self.aff[:, 0, 1] = S[:,1]*S[:,0].sin()
		self.aff[:, 1, 0] = S[:,1]*(-S[:,0]).sin()
		self.aff[:, 1, 1] = S[:,1]*S[:,0].cos()
		self.aff[:, 0, 2] = S[:,2]
		self.aff[:, 1, 2] = S[:,3]

		#print self.aff[0], self.samples[0], S[0]

		affineGrid = F.affine_grid(self.aff,images.size())

		placedMask = F.grid_sample(self.placedMask,affineGrid,padding_mode='border')
		maskedSticker = F.grid_sample(maskedSticker,affineGrid)

		maskedImage = torch.mul(images,placedMask)
		stickered = maskedSticker + maskedImage

		return stickered


if __name__ == "__main__":
	model = torch.load("mnist.pickle")
	mnist = MNIST()
	batch_size = 100
	trainloader = mnist.training(batch_size)
	model.cpu()

	mask = np.ones((1,9,9),dtype=np.float32)
	masker = AffineMaskSticker(mask,0.5,0.1,(1,28,28),90,0.6,0.7)
	masker.setBatchSize(batch_size)

	model.cuda()
	masker.cuda()

	targetLabel = torch.full([batch_size],9,dtype=torch.long)
	targetLabel = targetLabel.cuda()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD([masker.sticker], lr=0.01, momentum=0.9)

	print testTargetedAttack(model,mnist.testing(batch_size),masker,targetLabel.cpu())

	for j in range(20):
		for i, data in enumerate(trainloader, 0):
			images, labels = data
			images = images.cuda()

			optimizer.zero_grad()
			stickered = masker(images)
			if (j == 0 or j == 19) and i == 0:
				imshow(stickered.clone().detach())
			output = model(stickered)

			loss = criterion(output, targetLabel)
			loss.backward()
			optimizer.step()
			torch.clamp(masker.sticker,0.1,0.99)

	print testTargetedAttack(model,mnist.testing(batch_size),masker,targetLabel.cpu())

	sticker = torch.mul(masker.sticker,masker.mask).permute(1,2,0).detach()
	sticker = sticker.cpu().numpy()
	sticker = np.clip(sticker,0,1)
	if sticker.shape[2] == 1:
		sticker.shape = (9,9)
		sticker = gray2rgb(sticker)
	print(sticker.shape)

	io.imsave("sticker.png",sticker)