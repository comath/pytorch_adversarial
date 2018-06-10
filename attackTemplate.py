import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

from utils import numpyImages, conditionalPad

class BaseAttack(nn.Module):
	@property
	def usesLabels(self):
		return True

	@property
	def target(self):
		return None

	def visualize(self,images, model = None, num = 25, filename=None):
		attackedImgs = self.forward(images)[:num]
		imgs = images[:num]
		images = numpyImages(imgs)

		if model is not None:
			outputs = model(attackedImgs)
			_, attackPredicted = torch.max(outputs.data, 1)
			if self.target is None:
				correctOutputs = model(imgs)
				_, predicted = torch.max(correctOutputs.data, 1)
				success = (predicted != attackPredicted)
			else:
				success = (self.target[:num] == attackPredicted)
			success = success.cpu()
			
			padded_attack = conditionalPad(success,attackedImgs)
			attackedImages = numpyImages(padded_attack, padding=0)

		else:
			attackedImages = numpyImages(attackedImgs)
			


		fig, axs = plt.subplots(ncols=2)
		axs[0].imshow(images)
		axs[0].xaxis.set_visible(False)
		axs[0].yaxis.set_visible(False)
		axs[1].imshow(attackedImages)
		axs[1].xaxis.set_visible(False)
		axs[1].yaxis.set_visible(False)

		if filename is None:
			plt.show()
		else:
			plt.savefig(filename,dpi=fig.dpi*4,bbox_inches='tight')

	def test(self,model,test_set,device = None):
		if device is None:
			device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		cpu = torch.device("cpu")
		correct = torch.zeros((1,))
		total = torch.zeros((1,))
		correct = correct.to(device)
		total = total.to(device)
		dataIterator = tqdm(enumerate(test_set, 0),total = len(test_set))
		dataIterator.set_description("untargeted success rate: %.5f" % 0)
		update_rate = 10

		for i,data in dataIterator:
			images, labels = data
			images = images.to(device)
			if self.usesLabels:
				labels = labels.to(device)
				attackedImgs = self.forward(images,labels)
			else:
				attackedImgs = self.forward(images)

			outputs = model(attackedImgs)
			_, attackPredicted = torch.max(outputs.data, 1)

			total += images.size(0)

			if self.target is None:
				correctOutputs = model(images)
				_, predicted = torch.max(correctOutputs.data, 1)
				correct += (attackPredicted != predicted).sum().item()
			else: 
				correct += (attackPredicted == self.target).sum().item()

			if i % update_rate == update_rate - 1:
				total, correct = total.cpu(), correct.cpu()
				if self.target is None:
					dataIterator.set_description(
						"untargeted success rate: %.5f" % (correct[0]/total[0]))
				else:
					dataIterator.set_description(
						"targeted success rate: %.5f" % (correct[0]/total[0]))
				total, correct = total.cuda(), correct.cuda()


		        
		correct, total = correct.to(cpu), total.to(cpu)
		return correct[0]/total[0]