from attacks.gradientAttacks import *

from attacks.targets.datasets import CIFAR10
from attacks.targets.datasets import MNIST
import torch

import os
import argparse


def main():
	prog = "gradient_attacker"
	descr = "Demonstrate gradient attacks"
	parser = argparse.ArgumentParser(prog=prog, description=descr)
	parser.add_argument("-m", "--model", type=str, default=None, required=True, help="Target Model Pickle")
	parser.add_argument("-d", "--directory", type=str, default=None, required=False, help="Base location of images")
	args = parser.parse_args()

	if args.directory is None:
		image_base = args.model.split('.')[0]
	else:
		image_base = args.directory

	if torch.cuda.is_available():
		print("Using GPU 0")
		device = torch.cuda()
	else:
		print("No GPU, using CPU")
		device = torch.cpu()
	cpu = torch.cpu()


	model = torch.load(args.model)
	dataset = model.dataset()()

	batch_size = 400

	loader = dataset.training(batch_size)
	criterion = nn.CrossEntropyLoss()
	model.to(device)

	for f in range(1,11):
		f = 0.1*f
		ga = GradientAttack(model,criterion,f)
		nga = NormalizedGradientAttack(model,criterion,f)
		gsa = GradientSignAttack(model,criterion,f)

		dataiter = iter(dataset.testing(batch_size))
		images, labels = dataiter.next()
		images = images.to(device)

		ga.visualize(images,model,filename=image_base + "_ga_%1.2f.png" % (f,), diff_multiply = 10)
		ga_successRate = ga.test(model,dataset.testing(batch_size))
		print("The success rate on %s with gradient attack with eps %1.2f is %0.4f" % (args.model, f,ga_successRate))

		nga.visualize(images,model,filename=image_base + "_nga_%1.2f.png" % (f,), diff_multiply = 10)
		nga_successRate = nga.test(model,dataset.testing(batch_size))
		print("The success rate on %s with normalized gradient attack eps %1.2f is %0.4f" % (args.model, f,nga_successRate))

		gsa.visualize(images,model,filename=image_base + "_gsm_%1.2f.png" % (f,), diff_multiply = 1)
		gsa_successRate = gsa.test(model,dataset.testing(batch_size))
		print("The success rate on %s with gradient sign attack eps %1.2f is %0.4f" % (args.model, f,gsa_successRate))

if __name__ == '__main__':
	main()