import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datasets import MNIST
from ..utils import *


class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5,padding = 2)
        self.conv2 = nn.Conv2d(16, 16, 5,padding = 2)
        self.conv3 = nn.Conv2d(16, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.res = []

        self.dropout = nn.AlphaDropout(p=0.25)
        self.fc1 = nn.Linear(8*12*12, 240)
        self.fc2 = nn.Linear(240, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 8*12*12)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def trainMNISTConvNet(device):
    net = MNISTConvNet()
    mnist = MNIST()
    batch_size = 120
    trainloader = mnist.training(batch_size)


    print('Training MNIST ConvNet Model')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    trainModel(net,trainloader,optimizer,criterion,15,device)
    
    net.eval()
    print('Finished Training, getting accuracy')
    testloader = mnist.testing()
    accuracy = testAccuracy(net,testloader)
    
    print('Saving as: mnistConvNet.pkl')
    torch.save(net,"mnistConvNet.pkl")
