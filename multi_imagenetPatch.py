from xerxes.utils import *
import torch.optim as optim

from skimage import io, transform, img_as_float
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from xerxes.patchAttack import *
from xerxes.patchAttack.dataParallelTrainer import *
from xerxes.utils import *
from xerxes.targets.datasets import IMAGENET

batch_size = 20
imgnet = IMAGENET()

#testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=40, pin_memory=True)
loader = imgnet.training(batch_size)
dataiter = iter(loader)
imgs, labels = dataiter.next()
print(imgs.mean())
print(imgs.min())
print(imgs.max())
print(imgs.var())

mask = np.ones((3,224,224),dtype=np.float32)

model1 = torchvision.models.vgg16_bn(pretrained=True)
model2 = torchvision.models.alexnet(pretrained=True)
model3 = torchvision.models.densenet169(pretrained=True)
model4 = torchvision.models.resnet50(pretrained=True)

model_test = torchvision.models.vgg19_bn(pretrained=True).cuda()

#os.mkdir("./SGD_20batch/")
for i in range(1,2):
	for j in range(2):
		learningRate = i *  0.3
		mask = np.ones((3,224,224),dtype=np.float32)


		sticker = AdversarialSticker(mask,0.5)
		placer = AffinePlacer(mask,(3,224,224),15,0.6,(0.3,1.0))
		stickerAttack = StickerAttack(sticker,placer,346)

		criterion = nn.CrossEntropyLoss()
		
		optimizer = optim.SGD([sticker.sticker], lr= learningRate)
		os.mkdir("./SGD_20batch/%.3f/"%(learningRate,))

		#untrainedError = stickerAttack.test(model2,loader)

		stickerTrainer = StickerTrainer(
				sticker,
					[model1,
					model2,
					model3,
					model4],
				[criterion,
				criterion,
				criterion,
				criterion],
				346,
				placer)

		stickerTrainer.train(loader,optimizer,1,targetModel = model_test,root="SGD_20batch/%.3f/"%(learningRate,))
		torch.save(sticker,"SGD_20batch/%.3f/sticker_sgd.pkl"%(learningRate,))
