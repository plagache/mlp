import numpy as np
import random
from dlf.tensor import Tensor
from dlf.optimizer import GD, get_parameters
from dlf.dataset import load_dataset, compute_accuracy
from dlf.nn import Linear
from dlf.plot import plot_series

X_train, Y_train, X_test, Y_test = load_dataset()


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

optimizer = GD(get_parameters(model), 0.002)


def log_loss(y, p):
    return -((y * (p).log() + (1 - y) * (1 - p).log()).MEAN())


validation_losses = []
train_losses = []
train_accuracies = []
validation_accuracies = []

steps = 2000
for step in range(steps):
    Y = Tensor(Y_train)
    P = model(Tensor(X_train))

    train_loss = log_loss(Y, P)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if (step + 1) % 10 == 0:
        Y_T = Tensor(Y_test)
        P_T = model(Tensor(X_test))

        validation_loss = log_loss(Y_T, P_T)

        validation_losses.append(float(validation_loss.data[0]))
        train_losses.append(float(train_loss.data[0]))

        train_accuracy = compute_accuracy(Y.data, P.data)
        validation_accuracy = compute_accuracy(Y_T.data, P_T.data)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        print(f"Step {step + 1}: loss = {train_loss.data}, log_loss = {validation_loss.data}, train_acc = {train_accuracy:2f}%, validation_acc = {validation_accuracy:2f}%")

plot_series([("train", train_accuracies), ("validation", validation_accuracies)], "Accuracy")
plot_series([("train", train_losses), ("validation", validation_losses)], "Loss")
