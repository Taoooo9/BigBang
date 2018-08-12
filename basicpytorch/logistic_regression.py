import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.], [0.], [1.], [1.]]))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)


    def forward(self, x):
        y_pre = F.sigmoid(self.linear(x))
        return y_pre


model = Model()
loss = torch.nn.BCELoss(size_average = True)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


for epoch in range(500):
    y_pre = model(x_data)
    l = loss(y_pre, y_data)
    print(epoch, l.data[0])

    optimizer.zero_grad()
    l.backward()
    optimizer.step()

hour_val = Variable(torch.Tensor([[1.0]]))
print('predict 1 hour', 1.0, model(hour_val).data[0][0] > 0.5)
hour_var = Variable(torch.Tensor([[7.0]]))
print("predict 7 hours", 7.0, model(hour_var).data[0][0] > 0.5)




