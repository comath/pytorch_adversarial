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

mask = np.ones((1,15,15),dtype=np.float32)
masker = AffineMaskSticker(9,mask,(1,28,28),90,0.6,(0.2,1.2))
masker.setBatchSize(batch_size)

cudaDevice = torch.device("cuda:0")

model2.to(cudaDevice)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([masker.sticker], lr=0.1)

testset = mnist.testing(batch_size)
untrainedError = masker.test(model1,testset)


stickerTrainer = StickerTrainer(masker)
stickerTrainer.addModel(model1)
#stickerTrainer.addModel(model2,cudaDevice)
stickerTrainer.train(loader,optimizer,criterion,15)

trainedError = masker.test(model2,testset)
print('Untrained success rate: %.5f, Trained success rate: %.5f'%(untrainedError,trainedError))

dataiter = iter(testset)
images, labels = dataiter.next()
masker.visualize(images,model1,filename="sticker_attack.png")
masker.visualize(images,model2,filename="sticker_attack.png")

sticker = (masker.sticker + 1)/2
sticker = torch.mul(sticker,masker.mask).permute(1,2,0).detach()
sticker = sticker.cpu().numpy()
sticker = np.clip(sticker,0,1)
if sticker.shape[2] == 1:
	sticker.shape = (15,15)
	sticker = gray2rgb(sticker)

io.imsave("sticker.png",sticker)