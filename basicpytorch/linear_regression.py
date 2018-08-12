import torch
import torch.nn as nn
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pre = self.linear(x)
        return y_pre


model = Model()

loss = nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(500):
    y_pre = model(x_data)
    l = loss(y_pre, y_data)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    print('epoch:', epoch, 'loss:', l.data[0])


hour_var = Variable(torch.Tensor([[4.0]]))
y_pre = model(hour_var)
print('predict (after training)', 4, model(hour_var).data[0][0])