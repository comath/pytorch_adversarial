import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision
import torchvision.transforms as transforms

from datasets import CIFAR10
from utils import *

class residual(nn.Module):
    def __init__(self,nxn,connections,padding):
        super(residual, self).__init__()
        self.conv1 = nn.Conv2d(connections, connections, nxn,padding = padding)
        self.bn1 = nn.BatchNorm2d(connections)
        self.conv2 = nn.Conv2d(connections, connections, nxn,padding = padding)
        self.bn2 = nn.BatchNorm2d(connections)
    def forward(self,x):
        y = F.relu(self.conv1(x))
        y = self.bn1(y)
        y = F.relu(self.conv2(y))
        y = self.bn2(y)
        return x + y

class CIFAR10ResNet(nn.Module):
    def __init__(self,n):
        super(CIFAR10ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.res1 = []
        for i in range(0,n):
            res = residual(3,16,1)
            self.add_module("residual_"+str(i),res)
            self.res1.append(res)
        self.conv2 = nn.Conv2d(16, 32, 3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.res2 = []
        for i in range(n,2*n):
            res = residual(3,32,1)
            self.add_module("residual_"+str(i),res)
            self.res2.append(res)
        self.conv3 = nn.Conv2d(32, 64, 3,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.res3 = []
        for i in range(2*n,3*n):
            res = residual(3,64,1)
            self.add_module("residual_"+str(i),res)
            self.res3.append(res)

        self.fc1 = nn.Linear(64*8*8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        for res in self.res1:
            x = res(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        for res in self.res2:
            x = res(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        for res in self.res3:
            x = res(x)

        x = x.view(-1, 64*8*8)
        x = self.fc1(x)
        return x

class CIFAR10VGG(nn.Module):
    def __init__(self):
        super(CIFAR10ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.res1 = []
        for i in range(0,5):
            res = residual(3,16,1)
            self.add_module("residual_"+str(i),res)
            self.res1.append(res)
        self.conv2 = nn.Conv2d(16, 32, 3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.res2 = []
        for i in range(5,10):
            res = residual(3,32,1)
            self.add_module("residual_"+str(i),res)
            self.res2.append(res)
        self.conv3 = nn.Conv2d(32, 64, 3,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.res3 = []
        for i in range(10,15):
            res = residual(3,64,1)
            self.add_module("residual_"+str(i),res)
            self.res3.append(res)

        self.fc1 = nn.Linear(64*8*8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        for res in self.res1:
            x = res(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        for res in self.res2:
            x = res(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        for res in self.res3:
            x = res(x)

        x = x.view(-1, 64*8*8)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    import torch.optim as optim
    net = CIFAR10ResNet(3)
    cifar = CIFAR10()
    batch_size = 120
    trainloader = cifar.training(batch_size)
    if torch.cuda.is_available():
        print("Using GPU 0")
        device = torch.device("cuda:0")
    else:
        print("No GPU, using CPU")
        device = torch.device("cpu")

    cpu = torch.device("cpu")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.002)

    for epoch in range(500):  # loop over the dataset multiple times
        epoch_loss = torch.zeros((1,))
        epoch_loss = epoch_loss.cuda()
        update_loss = torch.zeros((1,))
        update_loss = update_loss.cuda()

        dataIterator = tqdm(enumerate(loader, 0),total = epoch_size)
        dataIterator.set_description("update loss: %.3f, epoch loss: %.3f" % (0,0))
        for i, data in dataIterator:
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            update_loss += loss
            if i % update_rate == update_rate - 1:    # print every 500 mini-batches
                epoch_loss += update_loss
                epoch_loss, update_loss = epoch_loss.cpu(), update_loss.cpu()
                dataIterator.set_description(
                    "update loss: %.3f, epoch loss: %.3f" % (
                        update_loss[0] / update_rate,
                        epoch_loss[0]/(i + 1),
                        ))
                update_loss.zero_()
                epoch_loss, update_loss = epoch_loss.cuda(), update_loss.cuda()

        epoch_loss = epoch_loss.cpu()
        print("Epoch %d/%d loss: %.4f" % (epoch+1,epochs,epoch_loss[0]/epoch_size))

    net.eval()
    print('Finished Training, getting accuracy')
    testloader = cifar.testing()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted = _.to(cpu), predicted.to(cpu)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    print('Saving as: cifarResNet.pickle')
    torch.save(net,"cifarResNet.pickle")
