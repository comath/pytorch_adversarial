from attacks.gradientAttacks import *

from attacks.targets.cifar10 import CIFAR10ResNet,residual
from attacks.targets.datasets import CIFAR10
model = torch.load("cifarnetbn.pickle")
dataset = CIFAR10()
batch_size = 400

loader = dataset.training(batch_size)
if torch.cuda.is_available():
	print("Using GPU 0")
	device = torch.device("cuda:0")
else:
	print("No GPU, using CPU")
	device = torch.device("cpu")
cpu = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
model.to(device)

successes = []
for f in range(1,11):
	f = 0.1*f
	gs = GradientSignAttack(model,criterion,f)

	dataiter = iter(dataset.testing(batch_size))
	images, labels = dataiter.next()
	images = images.cuda()
	gs.visualize(images,model,filename="gsm_%1.2f.png" % (f,), diff_multiply = 10)

	successRate = gs.test(model,dataset.testing(batch_size))
	successes.append(successRate)
	print("The success rate on a ResNet CIFAR with gradient sign eps %1.2f is %0.4f" % (f,successRate))

print successes