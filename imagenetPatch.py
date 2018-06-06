from utils import *
import torch.optim as optim

from skimage import io, transform, img_as_float
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, utils, datasets

from patchAttack import AffineMaskSticker

batch_size = 20

transform = transforms.Compose(
    [transforms.RandomAffine(36, translate=(0.1,0.1), scale=(0.9,1.1), shear=20),
    transforms.Resize((224,224)),
    transforms.ToTensor()])
trainset = datasets.ImageFolder("/home/sven/bulk/Data/ILSVRC/Data/DET/",transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=10)
dataiter = iter(loader)

masker = AffineMaskSticker("ai_mask.png",(3,224,224),90,0.6,(0.2,1.5))
masker.setBatchSize(batch_size)

targetLabel = torch.full([batch_size],346,dtype=torch.long)
shortTargetLabel = torch.full([1],346,dtype=torch.long)
targetLabel = targetLabel.cuda()

model = torchvision.models.resnet50(pretrained=True)

model.cuda()
masker.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([masker.sticker], lr=0.001)
#optimizer = optim.SGD([masker.sticker], lr=0.1, momentum=0.9)


print testTargetedAttack(model,loader,masker,targetLabel.cpu())

for epoch in range(100):
	running_loss = 0.0
	for i, data in enumerate(loader, 0):
		images, labels = data
		images = images.cuda()

		optimizer.zero_grad()
		stickered = masker(images)

		output = model(stickered)
		loss = criterion(output, targetLabel)
		loss.backward()
		# print statistics
		running_loss += loss.item()
		if i % 200 == 199:    # print every 500 mini-batches
			print('[%d, %5d] loss: %.3f' %
			  (epoch + 1, batch_size*(i + 1), running_loss / 2000))
			running_loss = 0.0
		optimizer.step()
		torch.clamp(masker.sticker,0.1,0.99)
	if (epoch == 0 or epoch == 9):
		imshow(stickered.clone().detach())

print testTargetedAttack(model,loader,masker,targetLabel.cpu())

sticker = torch.mul(masker.sticker,masker.mask).permute(1,2,0).detach()
sticker = sticker.cpu().numpy()
sticker = np.clip(sticker,0,1)
if sticker.shape[2] == 1:
	sticker.shape = (15,15)
	sticker = gray2rgb(sticker)
print(sticker.shape)

io.imsave("ai_sticker.png",sticker)