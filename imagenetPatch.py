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

from attacks.patchAttack import AffineMaskSticker, trainPatch



batch_size = 75

transform = transforms.Compose(
    [transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1)),
    transforms.Resize((224,224)),
    transforms.ToTensor()])
trainset = datasets.ImageFolder("/home/sven/data/ILSVRC/Data/DET/train/ILSVRC2013_train/",transform=transform)
testset = datasets.ImageFolder("/home/sven/data/ILSVRC/Data/DET/ver/",transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=40, pin_memory=True, drop_last=True)
loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=40, shuffle=True, pin_memory=True, drop_last=True)
dataiter = iter(loader)

masker = AffineMaskSticker("mask.png",(3,224,224),90,0.6,(0.3,2))
masker.setBatchSize(batch_size)

targetLabel = torch.full([batch_size],346,dtype=torch.long)
shortTargetLabel = torch.full([1],346,dtype=torch.long)
targetLabel = targetLabel.cuda()

model = torchvision.models.resnet50(pretrained=True)

model.cuda()
masker.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([masker.sticker], lr=0.0006)

#optimizer = optim.SGD([masker.sticker], lr=0.1, momentum=0.9)

#untrainedError = testTargetedAttack(model,testloader,masker,targetLabel.cpu())
#print('Untrained error: %.5f'%(untrainedError))
trainPatch(masker,model,loader,targetLabel,optimizer,criterion,10,batch_size)

#trainedError = testTargetedAttack(model,testloader,masker,targetLabel.cpu())
#print('Untrained error: %.5f, Trained error: %.5f'%(untrainedError,trainedError))

sticker = (masker.sticker + 1)/2
sticker = torch.mul(sticker,masker.mask).permute(1,2,0).detach()
sticker = sticker.cpu().numpy()
sticker = np.clip(sticker,0,1)
if sticker.shape[2] == 1:
	sticker.shape = (15,15)
	sticker = gray2rgb(sticker)

io.imsave("ai_sticker.png",sticker)
