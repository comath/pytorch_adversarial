import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

from utils import numpyImages, conditionalPad

class BaseAttack(nn.Module):
	'''
	Base attack class to be extended. This holds a visualize and test method. 

	Extend this and at least implement forward. You can also implement
	'''

	def forward(self):
		raise NotImplementedError("Please implement attack!")
	# Override this if the attack does not use labels
	@property
	def usesLabels(self):
		return True
	# Override this if the attack is targeted
	@property
	def target(self):
		return None


	def visualize(self,images, model = None, num = 25, diff_multiply = 0, filename=None):
		"""
		Shows a grid of images before and after the attack. If you supply a diff_multiply it 
		also shows the difference between the two images multiplied by what you provide

		Args:
			images: a set of images 
	        model: If passed will highlight in green which images we successfully misclassified
			num: the number of images to show, default 25
	        diff_multiply: If passed and > 0 will multiply the difference between the normal and 
	        	attacked images. Default 0
	        filename: If passed will save to that file
		"""
		attackedImages_torch = self.forward(images)[:num]
		images_torch = images[:num]
		images = numpyImages(images_torch)

		if model is not None:
			outputs = model(attackedImages_torch)
			_, attackPredicted = torch.max(outputs.data, 1)
			if self.target is None:
				correctOutputs = model(images_torch)
				_, predicted = torch.max(correctOutputs.data, 1)
				success = (predicted != attackPredicted)
			else:
				success = (self.target[:num] == attackPredicted)
			success = success.cpu()
			
			# Pad with green for a successful attack, red for unsuccessful
			padded_attack = conditionalPad(success,attackedImages_torch)
			attackedImages = numpyImages(padded_attack, padding=0)

		else:
			attackedImages = numpyImages(attackedImages_torch)
		if diff_multiply > 0:
			fig, axs = plt.subplots(ncols=3)
		else:
			fig, axs = plt.subplots(ncols=2)

		axs[0].imshow(images)
		axs[0].xaxis.set_visible(False)
		axs[0].yaxis.set_visible(False)
		axs[1].imshow(attackedImages)
		axs[1].xaxis.set_visible(False)
		axs[1].yaxis.set_visible(False)

		if diff_multiply > 0:
			x_diff = diff_multiply*(images_torch - attackedImages_torch)
			x_diff = numpyImages(x_diff)
			axs[2].imshow(x_diff)
			axs[2].xaxis.set_visible(False)
			axs[2].yaxis.set_visible(False)


		if filename is None:
				plt.show()
		else:
			plt.savefig(filename,dpi=fig.dpi*4,bbox_inches='tight')

	def test(self,model,test_set):
		"""
		Computes the success rate of the attack against the supplied model over the given test set.
		Pass the same model as the attack was designed for for a whitebox, or a different model for
		a black box. 

		Args:
			model: A model that can proccess the test set
	        test_set: A torch.utils.data.DataLoader
		"""

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
				total, correct = total.to(device), correct.to(device)


		        
		correct, total = correct.to(cpu), total.to(cpu)
		return correct[0]/total[0]