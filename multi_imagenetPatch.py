from xerxes.utils import *
import torch.optim as optim

from skimage import io, transform, img_as_float
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from xerxes.patchAttack import *
from xerxes.patchAttack.Trainer import *
from xerxes.utils import *
from xerxes.targets.datasets import IMAGENET

batch_size = 50
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

os.mkdir("./SGD/")
for i in range(1,6):
	for j in range(2):
		learningRate = i *  1.0
		mask = np.ones((3,224,224),dtype=np.float32)


		sticker = AdversarialSticker(mask,0.5)
		placer = AffinePlacer(mask,(3,224,224),15,0.6,(0.1,0.8))
		stickerAttack = StickerAttack(sticker,placer,346)

		criterion = nn.CrossEntropyLoss()
		if j > 0:
			optimizer = optim.SGD([sticker.sticker], lr= learningRate, weight_decay = 0.001)
			os.mkdir("./SGD/%.3f_wd0.001/"%(learningRate,))
		else:
			optimizer = optim.SGD([sticker.sticker], lr= learningRate)
			os.mkdir("./SGD/%.3f/"%(learningRate,))

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

		if j > 0:
			stickerTrainer.train(loader,optimizer,1,targetModel = model_test,root="SGD/%.3f_wd0.001/"%(learningRate,))
			torch.save(sticker,"SGD/%.3f_wd0.001/sticker_sgd.pkl"%(learningRate,))
		else:
			stickerTrainer.train(loader,optimizer,1,targetModel = model_test,root="SGD/%.3f/"%(learningRate,))
			torch.save(sticker,"SGD/%.3f/sticker_sgd.pkl"%(learningRate,))
