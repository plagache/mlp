import numpy as np
import random
from dlf.tensor import Tensor
from dlf.optimizer import SGD
from dlf.dataset import load_dataset
from dlf.nn import Linear


class Network:
    def __init__(self):
        self.l1 = Linear(30, 10)
        self.l2 = Linear(10, 10)
        self.l3 = Linear(10, 10)
        self.l4 = Linear(10, 2)

    def __call__(self, x: Tensor):
        x = self.l1(x).RELU()
        x = self.l2(x).RELU()
        x = self.l3(x).RELU()
        x = self.l4(x).RELU()
        return x.SOFTMAX()


model = Network()

X_train, Y_train, X_test, Y_test = load_dataset()

steps = 5

optimizer = SGD([model.l1.weight, model.l2.weight], 0.002)
# optimizer = SGD([model.l1.weight, model.l1.bias, model.l2.weight, model.l2.bias], 0.002)


# def accuracy(true, prediction):
#     true = np.argmax(true.to_numpy(), axis=-1)
#     prediction = np.argmax(prediction.data, axis=-1)
#     return (true == prediction).mean()


def loss(y, p):
    return -(y * p.log() + (1 - y) * (1 - p).log()).mean()


for step in range(steps):
    print(step)

    Y = Tensor(Y_train)
    P = model(Tensor(X_train))

    # precision = accuracy(Y_train, P)
    # print(f"{precision * 100}%")

    loss_val = loss(Y, P)
    print(f"{loss_val.data=}")
    print(f"{P.data=}")

    loss_val.backward()

    optimizer.step()
    print(model.l1.weight)
