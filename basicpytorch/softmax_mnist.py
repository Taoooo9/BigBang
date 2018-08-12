import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable


root = './mnist_data/'

train_dataset = datasets.MNIST(root = root,
                            train = True, transform = transforms.ToTensor(),
                            download = True)


test_dataset = datasets.MNIST(root = root,
                           train = False, transform = transforms.ToTensor())


batch_size = 64

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size,
                          shuffle = True)

test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size,
                         shuffle = False)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


model = Model()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)

def train(epoch):
    model.train()
    for index, (data, target) in enumerate(train_loader):
        print("index:", index)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        l = loss(output, target)
        l.backward()
        optimizer.step()
        if index % 10 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, index * len(data), len(train_loader.dataset),
                100. * index / len(train_loader), l.data[0]
            ))

print('train_loader.dataset:', len(train_loader.dataset), 'train_loader:', len(train_loader), 'train_dataset', len(train_dataset))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile = True), Variable(target)
        output = model(data)
        test_loss += loss(output, target).data[0]
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()




