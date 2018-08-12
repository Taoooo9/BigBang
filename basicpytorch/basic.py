import numpy as np
import matplotlib.pyplot as pyplot

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    return x * w

def loss(x, y):
    pre_y = forward(x)
    return (pre_y - y)*(pre_y - y)


w_list = []
loss_list = []


for w in np.arange(0, 4.1, 0.1):
    print('w:', w)
    l_sum = 0
    for x, y in zip(x_data, y_data):
        pre_y = forward(x)
        l = loss(x, y)
        l_sum += l
        print("x:", x, 'y:', y, 'pre_y:', pre_y, 'loss:', l)
    print('mse=', l_sum / 3)
    w_list.append(w)
    loss_list.append(l_sum / 3)

pyplot.plot(w_list, loss_list)
pyplot.ylabel('loss')
pyplot.xlabel('w')
pyplot.show()



