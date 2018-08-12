import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]), requires_grad = True)

def forward(x):
    return x * w


def loss(x, y):
    y_pre = forward(x)
    return (y_pre - y) * (y_pre - y)


print('first:', 4, forward(4).data[0])


for i in range(10):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('W:', w.grad.data[0])
        w.data -= w.grad.data * 0.01
        w.grad.data.zero_()
    print('epoch:', i, 'loss', l.data[0])

print('end:', 4, forward(4).data[0])

