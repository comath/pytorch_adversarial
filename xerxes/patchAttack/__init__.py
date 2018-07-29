import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from tqdm import tqdm
from ..attackTemplate import BaseAttack
from skimage import io, transform, img_as_float
from ..utils import *

class AdversarialSticker(nn.Module):
	def __init__(self,mask,mean,pixel_range,std = 0.1):
		super(AdversarialSticker, self).__init__()

		if isinstance(mask, str):
			from skimage import io, img_as_float
			mask = img_as_float(io.imread(mask))
			mask = mask.transpose(2,0,1)

		self.mask = torch.from_numpy(mask)
		self.mask = self.mask.type(torch.FloatTensor)
		self.mask = nn.Parameter(self.mask,requires_grad = False)

		self.mean = mean
		self.pixel_range = pixel_range
		zero_mean = torch.zeros(self.mask.size())
		std = std*torch.ones(self.mask.size())
		self.sticker = nn.Parameter(torch.normal(zero_mean,std))

	def forward(self):
		maskedSticker = self.mean + self.sticker 						# place it in the middle of the pixel space
		maskedSticker = torch.mul(maskedSticker,self.mask)
		return maskedSticker

	def clamp(self):
		self.sticker.data = torch.clamp(self.sticker,self.pixel_range[0],self.pixel_range[1]).data
		'''
		centered = self.sticker
		m = centered.abs().max()
		if m > 0.5:
			self.sticker.data = ((0.5/m)*self.sticker).data
		'''
	def size(self):
		return self.mask.size()

	def save(self,filename):
		sticker = self.__call__()
		sticker = sticker.permute(1,2,0).detach()
		sticker = sticker.cpu().numpy()
		sticker = np.clip((sticker-self.pixel_range[0])/(self.pixel_range[1]-self.pixel_range[0]),0,1)
		io.imsave(filename,sticker)

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