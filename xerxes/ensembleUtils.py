import torch.nn as nn
import torch

class targetModelWrap(nn.Module):
	def __init__(self,model,target,device = None):
		super(targetModelWrap, self).__init__()
		self.model = model
		self.target = target
		self.batchSize = 0
		self.targetLabel = None
		self.device = device
		self.numDev = torch.cuda.device_count()
		if self.device is None:
			self.models = nn.parallel.replicate(model.cuda(), list(range(self.numDev)))

	def __setBatchSize__(self,batchSize):
		self.batchSize = batchSize
		if self.targetLabel is not None:
			del self.targetLabel
		self.targetLabel = torch.full([self.batchSize],
				self.target,
				dtype=torch.long)
		if self.device is None:
			self.targetLabel = self.targetLabel.cuda()
			#self.targetLabel = torch.cuda.comm.scatter(self.targetLabel,list(range(self.numDev)))
		else:
			self.targetLabel = self.targetLabel.to(self.device)

	def test(self,images):
		total = images.size(0)
		if self.batchSize != images.size()[0]:
			self.__setBatchSize__(images.size()[0])
		if self.device is None:
			images = images.cuda()
			images = zip(torch.cuda.comm.scatter(images,list(range(self.numDev))))
			preds = nn.parallel.parallel_apply(self.models,images)
			y = torch.cuda.comm.gather(preds,destination = 0)
			val, pred = torch.max(y.data, 1)
			correct = ((pred == self.targetLabel).sum().item())
		else:
			images = images.to(self.device)
			y = self.model(images)
			val, pred = torch.max(y.data, 1)
			correct = ((pred == self.targetLabel).sum().item())

		return total,correct

