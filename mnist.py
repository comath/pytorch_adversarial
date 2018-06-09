import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datasets import MNIST
from utils import *


class MNISTNet(nn.Module):
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
        if self.training:
            x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    net = MNISTNet()
    mnist = MNIST()
    batch_size = 400
    trainloader = mnist.training(batch_size)

    # Check for compatible GPU, then moves the model to it
    if torch.cuda.is_available():
        print("Using GPU 0")
        device = torch.device("cuda:0")
    else:
        print("No GPU, using CPU")
        device = torch.device("cpu")
    cpu = torch.device("cpu")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(25):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
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
            running_loss += loss.item()
            if i % 100 == 99:    # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_size*(i + 1), running_loss / 2000))
                running_loss = 0.0

    net.eval()
    print('Finished Training, getting accuracy')
    testloader = mnist.testing()
    accuracy = testAccuracy(net,testloader)
    print('Accuracy of the network on the %d test images: %d %%' % (
        len(testloader),
        100 * accuracy))
    print('Saving as: mnist.pickle')
    net.cpu()
    torch.save(net,"mnist.pickle")
