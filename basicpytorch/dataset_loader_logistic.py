import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.utils.data import  DataLoader, Dataset


class DiabetesDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./data/diabetes.csv.gz', delimiter = ','
                        , dtype = np.float32)
        self.len = xy.shape[0]
        self.x = torch.from_numpy(xy[:, 0:-1])
        self.y = torch.from_numpy(xy[:, [0]])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset()
data_loader = DataLoader(dataset = dataset, shuffle = True,
                         batch_size = 64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.sigmoid(self.l1(x))
        y = self.sigmoid(self.l2(y))
        y = self.sigmoid(self.l3(y))
        return y

model = Model()

loss = nn.BCELoss(size_average = True)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

for epoch in range(2):
    for i, data in enumerate(data_loader):
        x_data, y_data = data

        inputs, labels = Variable(x_data), Variable(y_data)

        y_pre = model(inputs)
        l = loss(y_pre, labels)

        print(epoch, i, 'loss:', l.data[0])

        optimizer.zero_grad()
        l.backward()
        optimizer.step()




