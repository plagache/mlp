import numpy as np
import random
from dlf.tensor import Tensor
from dlf.optimizer import SGD
from dlf.dataset import load_dataset, compute_accuracy
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
        x = self.l4(x).SOFTMAX()
        return x


model = Network()

X_train, Y_train, X_test, Y_test = load_dataset()

steps = 1000

optimizer = SGD([model.l1.weight, model.l1.bias, model.l2.weight, model.l2.bias, model.l3.weight, model.l3.bias, model.l4.weight, model.l4.bias], 0.00002)


def loss(y, p):
    weighted_log = y * p.log()
    per_sample_loss = weighted_log.SUM()
    return -per_sample_loss.MEAN()


for step in range(steps):
    Y = Tensor(Y_train)
    P = model(Tensor(X_train))

    loss_val = loss(Y, P)

    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()

    if (step + 1) % 10 == 0:
        train_accuracy = compute_accuracy(Y.data, P.data)
        validation_accuracy = compute_accuracy(Y_test, model(Tensor(X_test)).data)
        print(f"Step {step + 1}: loss = {loss_val.data}, train_acc = {train_accuracy:2f}%, validation_acc = {validation_accuracy:2f}%")
