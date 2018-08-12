import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
Y = np.array([1, 0, 0])

Y_pred1 = np.array([0.7, 0.2, 0.1])
Y_pred2 = np.array([0.1, 0.3, 0.6])

print('first--loss1:', np.sum(-Y * np.log(Y_pred1)))
print('first--loss2:', np.sum(-Y * np.sum(Y_pred2)))


loss = nn.CrossEntropyLoss()


Y = Variable(torch.LongTensor([0]), requires_grad = False)

Y_pred1 = Variable(torch.Tensor([[2.0, 1.0, 0.1]]))
Y_pred2 = Variable(torch.Tensor([[0.5, 2.0, 0.3]]))

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print('2nd--loss1:', l1.data)
print('2nd--loss2:', l2.data)

print('Y_pred1:', torch.max(Y_pred1.data, 1)[1])
print('Y_pred2:', torch.max(Y_pred2.data, 1)[1])


Y = Variable(torch.LongTensor([2, 0, 1]), requires_grad=False)

Y_pred1 = Variable(torch.Tensor([[0.1, 0.2, 0.9],
                                 [1.1, 0.1, 0.2],
                                 [0.2, 2.1, 0.1]]))


Y_pred2 = Variable(torch.Tensor([[0.8, 0.2, 0.3],
                                 [0.2, 0.3, 0.5],
                                 [0.2, 0.2, 0.5]]))

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print("Batch Loss1 = ", l1.data, "\nBatch Loss2=", l2.data)

