import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch

def imshow(imgs):
	images = torchvision.utils.make_grid(imgs)
	images = images / 2 + 0.5     # unnormalize
	npimages = images.cpu().numpy()
	images = np.transpose(npimages, (1, 2, 0))
	images = np.clip(images,0,1)
	plt.figure()
	# show images
	plt.imshow(images)
	plt.show()

def testAccuracy(model,test_set,device = None):
	if device is None:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cpu = torch.device("cpu")

	correct = 0.0
	total = 0.0
	with torch.no_grad():
		for data in test_set:
			images, labels = data
			images = images.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			_, predicted = _.to(cpu), predicted.to(cpu)

			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	        
	return correct/total

def testTargetedAttack(model,test_set,attack, target,device = None):
	if device is None:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cpu = torch.device("cpu")
	correct = 0.0
	total = 0.0
	with torch.no_grad():
		for data in test_set:
			images, labels = data
			images = images.to(device)
			attackedImgs = attack(images)
			outputs = model(attackedImgs)
			_, predicted = torch.max(outputs.data, 1)
			_, predicted = _.to(cpu), predicted.to(cpu)

			total += target.size(0)
			correct += (predicted == target).sum().item()
	        
	return correct/total

def testNonTargetedAttack(model,test_set,attack, target,device = None):
	if device is None:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cpu = torch.device("cpu")
	correct = 0.0
	total = 0.0
	with torch.no_grad():
		for data in test_set:
			images, labels = data
			images = images.to(device)
			attackedImgs = attack(images)
			outputs = model(attackedImgs)
			_, predicted = torch.max(outputs.data, 1)
			_, predicted = _.to(cpu), predicted.to(cpu)

			total += labels.size(0)
			correct += (predicted != labels).sum().item()
	        
	return correct/total