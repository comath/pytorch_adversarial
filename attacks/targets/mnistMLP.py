import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datasets import MNIST
from ..utils import *

class MNISTMLP(nn.Module):
    def __init__(self):
        super(MNISTMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(784, 300),
            nn.ReLU(True),
            nn.Linear(300, 150),
            nn.ReLU(True),
            nn.Linear(150, 10))

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.mlp(x)

    def dataset(self):
            return MNIST

def trainMNISTMLP(device=None,directory = ''):
    if device is None:
        device = getDevice()

    net = MNISTMLP()
    mnist = MNIST()
    batch_size = 120
    trainloader = mnist.training(batch_size)

    print('Training MNIST MLP Model')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    trainModel(net,trainloader,optimizer,criterion,5,device)
    
    net.eval()
    print('Finished Training, getting accuracy')
    testloader = mnist.testing()
    accuracy = testAccuracy(net,testloader)
    
    model_path = os.path.join(directory, "mnistMLP.pkl")
    print('Saving as: mnistMLP.pkl')
    torch.save(net,model_path)