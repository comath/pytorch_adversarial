from xerxes.targets.mnistConvNet import MNISTConvNet
from xerxes.targets.mnistMLP import MNISTMLP
from xerxes.targets.datasets import MNIST
from xerxes.utils import *
from xerxes.patchAttack import *
import torch.optim as optim
from skimage import io, transform, img_as_float
from skimage.color import gray2rgb


model = torch.load("mnistMLP.pkl")
mnist = MNIST()
batch_size = 400
loader = mnist.training(batch_size)
model.cpu()

mask = np.ones((3,15,15),dtype=np.float32)
masker = AffineMaskSticker(9,mask,(3,32,32),90,0.6,(0.2,1.2))
masker.setBatchSize(batch_size)

model.cuda()
masker.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([masker.sticker], lr=0.001)

testset = mnist.testing(batch_size)
untrainedError = masker.test(model,testset)


trainPatch(masker,model,loader,optimizer,criterion,5,batch_size)

trainedError = masker.test(model,testset)
print('Untrained success rate: %.5f, Trained success rate: %.5f'%(untrainedError,trainedError))

dataiter = iter(testset)
images, labels = dataiter.next()
images = images.cuda()
masker.visualize(images,model,filename="sticker_attack.png")

sticker = (masker.sticker + 1)/2
sticker = torch.mul(sticker,masker.mask).permute(1,2,0).detach()
sticker = sticker.cpu().numpy()
sticker = np.clip(sticker,0,1)
if sticker.shape[2] == 1:
	sticker.shape = (15,15)
	sticker = gray2rgb(sticker)

io.imsave("sticker.png",sticker)