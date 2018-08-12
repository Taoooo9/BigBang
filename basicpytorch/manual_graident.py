x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0



def forward(x):
    return x * w

def loss(x, y):
    y_pre = forward(x)
    return (y_pre - y) * (y_pre - y)


def gradient(x, y):  # d loss / d w
    return 2 * x * (x * w - y)

print("predict (before training)", 4, forward(4))

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= grad * 0.01
        l = loss(x, y)
        print("\tgrad: ", x, y, round(grad, 2))
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))


print("predict (after training)",  "4 hours", forward(4))



