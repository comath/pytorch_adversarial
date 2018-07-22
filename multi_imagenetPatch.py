from xerxes.utils import *
import torch.optim as optim

from skimage import io, transform, img_as_float
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from xerxes.patchAttack import *
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

model1 = torchvision.models.resnet101(pretrained=True)
model2 = torchvision.models.resnet50(pretrained=True)

sticker = AdversarialSticker(mask,0.5)
placer = AffinePlacer(mask,(3,224,224),90,0.6,(0.,1.0))
stickerAttack = StickerAttack(sticker,placer,346)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([sticker.sticker], lr= 0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

untrainedError = stickerAttack.test(model2,loader)

stickerTrainer = StickerTrainer(stickerAttack,[model1,model2],[criterion,criterion])
stickerTrainer.train(loader,optimizer,10)
torch.save(sticker,"sticker_sgd.pkl")

model2 = model2.cuda(0)
trainedError = stickerAttack.test(model2,loader)
print('Untrained success rate: %.5f, Trained success rate: %.5f'%(untrainedError,trainedError))
stickerAttack = stickerAttack.cuda(0)
dataiter = iter(loader)
images, labels = dataiter.next()
images = images.cuda(0)
stickerAttack.visualize(images,model1,filename="sticker_attack1.png")
model2 = model2.cuda(0)
stickerAttack.visualize(images,model2,filename="sticker_attack2.png")


sticker = (sticker() + 1)/2
sticker = sticker.permute(1,2,0).detach()
sticker = sticker.cpu().numpy()
sticker = np.clip(sticker,0,1)
io.imsave("ai_sticker_sgd_%f_res101_50_v2.png"%(trainedError),sticker)
