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

for f in range(1,11):
	f = 0.1*f
	ga = GradientAttack(model,criterion,f)
	nga = NormalizedGradientAttack(model,criterion,f)
	gsa = GradientSignAttack(model,criterion,f)

	dataiter = iter(dataset.testing(batch_size))
	images, labels = dataiter.next()
	images = images.cuda()

	ga.visualize(images,model,filename="gsm_%1.2f.png" % (f,), diff_multiply = 10)
	ga_successRate = ga.test(model,dataset.testing(batch_size))
	print("The success rate on a ResNet CIFAR with gradient attack with eps %1.2f is %0.4f" % (f,ga_successRate))

	nga.visualize(images,model,filename="gsm_%1.2f.png" % (f,), diff_multiply = 10)
	nga_successRate = nga.test(model,dataset.testing(batch_size))
	print("The success rate on a ResNet CIFAR with normalized gradient attack eps %1.2f is %0.4f" % (f,nga_successRate))

	gsa.visualize(images,model,filename="gsm_%1.2f.png" % (f,), diff_multiply = 10)
	gsa_successRate = gsa.test(model,dataset.testing(batch_size))
	print("The success rate on a ResNet CIFAR with gradient sign attack eps %1.2f is %0.4f" % (f,gsa_successRate))
