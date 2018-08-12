import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms



root = './data/'
batch_size = 64

train_dataset = datasets.MNIST(root = root,
                               transform = transforms.ToTensor(),
                               train = True,
                               download = True)

test_dataset = datasets.MNIST(root = root,
                              transform = transforms.ToTensor(),
                              train = False)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size,
                          shuffle = True)

test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size,
                         shuffle = False)


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)
        x = self.fc(x)
        x = F.log_softmax(x)
        return x


model = CNN()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)



def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        l = F.nll_loss(output, target)
        l.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), l.data[0]))


def test():
    model.eval()
    correct = 0
    test_loss = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average = False).data[0]
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average Loss: {:.4f} , Accuracy{}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 11):
    train(epoch)
    test()

