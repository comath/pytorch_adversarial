from attacks.utils import *
import torch.optim as optim

from skimage import io, transform, img_as_float
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.patchAttack import AffineMaskSticker, trainPatch
from attacks.targets.datasets import IMAGENET



batch_size = 60
imgnet = IMAGENET()

#testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=40, pin_memory=True)
loader = imgnet.training(batch_size)
dataiter = iter(loader)
imgs, labels = dataiter.next()
print(imgs.mean())
print(imgs.var())

mask = np.ones((3,224,224),dtype=np.float32)

masker = AffineMaskSticker(346,mask,(3,224,224),90,0.6,(0.3,1.2),mean= 0.5)
masker.setBatchSize(batch_size)

model = torchvision.models.resnet101(pretrained=True)

model.cuda()
masker.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([masker.sticker], lr= 0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
#optimizer = optim.SGD([masker.sticker], lr=0.1, momentum=0.9)

#untrainedError = testTargetedAttack(model,testloader,masker,targetLabel.cpu())
#print('Untrained error: %.5f'%(untrainedError))
trainPatch(masker,model,loader,optimizer,criterion,5,batch_size)
images, labels = dataiter.next()
images = images.cuda()
masker.visualize(images,model,filename="sticker_attack.png")

#trainedError = testTargetedAttack(model,testloader,masker,targetLabel.cpu())
#print('Untrained error: %.5f, Trained error: %.5f'%(untrainedError,trainedError))

sticker = torch.mul(masker.sticker,masker.mask).permute(1,2,0).detach()
sticker = sticker.cpu().numpy() + 0.5
sticker = np.clip(sticker,0,1)
if sticker.shape[2] == 1:
	sticker.shape = (15,15)
	sticker = gray2rgb(sticker)

io.imsave("ai_sticker_vgg19.png",sticker)
