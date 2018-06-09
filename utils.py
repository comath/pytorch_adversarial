import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
from tqdm import tqdm

def numpyImages(imgs):
	images = torchvision.utils.make_grid(imgs)
	images = images / 2 + 0.5     # unnormalize
	npimages = images.cpu().numpy()
	images = np.transpose(npimages, (1, 2, 0))
	return np.clip(images,0,1)

def imshow(imgs):
	images = numpyImages(imgs)
	plt.figure()
	# show images
	plt.imshow(images)
	plt.show()

def visualizeAttack(images, attack, model = None):
	images = numpyImages(images)
	attackImages = numpyImages(attack(images))
	fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)




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
	correct = torch.zeros((1,))
	total = torch.zeros((1,))
	target = target.to(device)
	correct = correct.to(device)
	total = total.to(device)
	dataIterator = tqdm(enumerate(test_set, 0),total = len(test_set))
	dataIterator.set_description("targeted success rate: %.5f" % 0)
	update_rate = 10

	for i,data in dataIterator:
		images, labels = data
		images = images.to(device)
		if attack.usesLabels:
			labels = labels.to(device)
			attackedImgs = attack(images,labels)
		else:
			attackedImgs = attack(images)

		
		outputs = model(attackedImgs)
		_, predicted = torch.max(outputs.data, 1)

		total += target.size(0)

		correct += (predicted == target).sum().item()

		if i % update_rate == update_rate - 1:
			total, correct = total.cpu(), correct.cpu()
			dataIterator.set_description(
				"targeted success rate: %.5f" % (correct[0]/total[0]))
			total, correct = total.cuda(), correct.cuda()

	        
	correct, total = correct.to(cpu), total.to(cpu)
	return correct[0]/total[0]

def testNonTargetedAttack(model,test_set,attack,device = None):
	if device is None:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cpu = torch.device("cpu")
	correct = torch.zeros((1,))
	total = torch.zeros((1,))
	correct = correct.to(device)
	total = total.to(device)
	dataIterator = tqdm(enumerate(test_set, 0),total = len(test_set))
	dataIterator.set_description("targeted success rate: %.5f" % 0)
	update_rate = 10

	for i,data in dataIterator:
		images, labels = data
		images = images.to(device)
		if attack.usesLabels:
			labels = labels.to(device)
			attackedImgs = attack(images,labels)
		else:
			attackedImgs = attack(images)

		outputs = model(attackedImgs)
		_, predicted = torch.max(outputs.data, 1)

		total += images.size(0)
		correct += (predicted != labels).sum().item()

		if i % update_rate == update_rate - 1:
			total, correct = total.cpu(), correct.cpu()
			dataIterator.set_description(
				"untargeted success rate: %.5f" % (correct[0]/total[0]))
			total, correct = total.cuda(), correct.cuda()

	        
	correct, total = correct.to(cpu), total.to(cpu)
	return correct[0]/total[0]