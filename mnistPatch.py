from xerxes.targets.mnistMLP import MNISTMLP
from xerxes.targets.mnistConvNet import MNISTConvNet
from xerxes.targets.datasets import MNIST
from xerxes.utils import *
from xerxes.patchAttack import *
import torch.optim as optim
from skimage import io, transform, img_as_float
from skimage.color import gray2rgb


model1 = torch.load("models/mnistMLP.pkl")
model2 = torch.load("models/mnistConvNet.pkl")
mnist = MNIST()
batch_size = 400
loader = mnist.training(batch_size)
testset = mnist.testing(batch_size)

mask = np.ones((1,15,15),dtype=np.float32)
sticker = AdversarialSticker(mask,0)
placer = AffinePlacer(mask,(1,28,28),90,0.6,(0.2,1.2))
stickerAttack = StickerAttack(sticker,placer,9)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([sticker.sticker], lr=0.1)
untrainedError = stickerAttack.test(model1,testset)

model1 = model1.cuda(0)
stickerAttack = stickerAttack.cuda(0)
model2 = model2.cuda(1)

stickerTrainer = StickerTrainer(stickerAttack,[model1,model2],[criterion,criterion])

import cProfile

stickerTrainer.train(loader,optimizer,15)


trainedError = stickerAttack.test(model2,testset)
print('Untrained success rate: %.5f, Trained success rate: %.5f'%(untrainedError,trainedError))
stickerAttack = stickerAttack.cuda(0)
dataiter = iter(testset)
images, labels = dataiter.next()
images = images.cuda(0)
stickerAttack.visualize(images,model1,filename="sticker_attack.png")
model2 = model2.cuda(0)
stickerAttack.visualize(images,model2,filename="sticker_attack.png")

sticker = (sticker() + 1)/2
sticker = sticker.permute(1,2,0).detach()
sticker = sticker.cpu().numpy()
sticker = np.clip(sticker,0,1)
if sticker.shape[2] == 1:
	sticker.shape = (15,15)
	sticker = gray2rgb(sticker)

io.imsave("ai_sticker_%f_mnistConv_mlp_v2.png"%(trainedError),sticker)
