import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

xy = np.loadtxt('./data/diabetes.csv.gz', delimiter = ',', dtype = np.float32)
x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[:, [-1]]))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_pre = self.sigmoid(self.l1(x))
        y_pre = self.sigmoid(self.l2(y_pre))
        y_pre = self.sigmoid(self.l3(y_pre))

        return y_pre


model = Model()

loss = nn.BCELoss(size_average = True)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

for epoch in range(100):
    y_pre = model(x_data)
    l = loss(y_pre, y_data)

    print(epoch, l.data[0][0])

    optimizer.zero_grad()
    l.backward()
    optimizer.step()


