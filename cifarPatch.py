from xerxes.targets.cifar10ResNet import CIFAR10ResNet,residual
from xerxes.targets.datasets import CIFAR10
from xerxes.utils import *
from xerxes.patchAttack import *
from xerxes.patchAttack.dataParallelTrainer import *
import torch.optim as optim
from skimage import io, transform, img_as_float
from skimage.color import gray2rgb


model1 = torch.load("models/cifar10ResNet22.pkl")
model2 = torch.load("models/cifar10VGG9.pkl")
model_test = torch.load("models/cifar10VGG12.pkl")
cifar = CIFAR10()
batch_size = 400
loader = cifar.training(batch_size)
model1.cpu()
model2.cpu()

mask = np.ones((3,32,32),dtype=np.float32)

sticker = AdversarialSticker(mask,0,(-1,1))
placer = AffinePlacer(mask,(3,32,32),15,0.6,(0.2,1.0))
stickerAttack = StickerAttack(sticker,placer,9)

testset = cifar.testing(batch_size)
untrainedError = stickerAttack.test(model_test,testset)

criterion = nn.CrossEntropyLoss()
		
optimizer = optim.SGD([sticker.sticker], lr= 0.3)

#untrainedError = stickerAttack.test(model2,loader)

stickerTrainer = StickerTrainer(
		sticker,
			[model1,
			model2],
		[criterion,
		criterion],
		9,
		placer)

stickerTrainer.train(loader,optimizer,1,targetModel = model_test)

dataiter = iter(testset)
images, labels = dataiter.next()
images = images.cuda()
stickerAttack.visualize(images,model_test,filename="sticker_attack.png")

sticker.save("cifar_sticker.png")