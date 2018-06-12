from attacks.utils import *
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



batch_size = 150

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=40, pin_memory=True)
loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=40, shuffle=True, pin_memory=True)
dataiter = iter(loader)
imgs, labels = dataiter.next()

masker = AffineMaskSticker(346,"mask.png",(3,224,224),90,0.6,(0.3,1.2),mean= 0.5)
masker.setBatchSize(batch_size)

model = torchvision.models.resnet18(pretrained=True)

model.cuda()
masker.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([masker.sticker], lr= 1, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#optimizer = optim.SGD([masker.sticker], lr=0.1, momentum=0.9)

#untrainedError = testTargetedAttack(model,testloader,masker,targetLabel.cpu())
#print('Untrained error: %.5f'%(untrainedError))
trainPatch(masker,model,loader,optimizer,criterion,1,batch_size)

#trainedError = testTargetedAttack(model,testloader,masker,targetLabel.cpu())
#print('Untrained error: %.5f, Trained error: %.5f'%(untrainedError,trainedError))

sticker = torch.mul(masker.sticker,masker.mask).permute(1,2,0).detach()
sticker = sticker.cpu().numpy()
sticker = np.clip(sticker,0,1)
if sticker.shape[2] == 1:
	sticker.shape = (15,15)
	sticker = gray2rgb(sticker)

io.imsave("ai_sticker_vgg19.png",sticker)
