import torch.nn as nn
import torch

class PatchModelWrap(nn.Module):
	'''
	Wraps a model so that we can easily hand it a patch and get back the gradient
	'''
	def __init__(self,model,loss,target,placerArgs):
		super(ModelGrad, self).__init__()
		self.model = model
		self.add_module("model",model)
		self.loss = loss
		self.target = target
		self.add_module("loss",loss)
		self.placer = AffinePlacer(*placerArgs)
		self.add_module("placer",res)
		self.updateLoss = nn.Parameter(torch.zeros(1),requires_grad=False)

	def __setBatchSize__(self,batchSize):
		self.batchSize = batchSize
		if self.targetLabel is not None:
			del self.targetLabel
		self.targetLabel = torch.full([self.batchSize],
				self.my_target,
				dtype=torch.long,
				device=self.updateLoss.device)

	def getRunningLoss(self):
		retVal = self.updateLoss
		self.updateLoss[0] = 0
		return self.updateLoss

	def forward(self,images, sticker):
		if self.batchSize != images.size()[0]:
			self.__setBatchSize__(images.size()[0])
		sticker_adv = sticker.requires_grad_()
		stickered = self.placer(images,sticker_adv)
		y = self.model(stickered)
		loss = self.lossfn(y,targetLabel)
		stickerGrad = torch.autograd.grad(loss, sticker_adv)[0]
		return stickerGrad,loss