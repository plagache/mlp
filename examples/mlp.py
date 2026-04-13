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


def loss(y, p):
    weighted_log = y * p.log()
    per_sample_loss = weighted_log.SUM()
    return -per_sample_loss.MEAN()


def log_loss(y, p):
    return -((y * (p).log() + (1 - y) * (1 - p).log()).MEAN())

train_accuracies = []
validation_accuracies = []
log_losses = []
losses = []
steps = 2000
y_shape = int
p_shape = int
for step in range(steps):
    Y = Tensor(Y_train)
    P = model(Tensor(X_train))

    loss_val = log_loss(Y, P)
    # loss_val = loss(Y, P)

    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()

    if (step + 1) % 10 == 0:
        Y_T = Tensor(Y_test)
        P_T = model(Tensor(X_test))
        log_loss_val = log_loss(Y_T, P_T)
        log_losses.append(log_loss_val.data)
        losses.append(loss_val.data)
        train_accuracy = compute_accuracy(Y.data, P.data)
        validation_accuracy = compute_accuracy(Y_T.data, P_T.data)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
        print(f"Step {step + 1}: loss = {loss_val.data}, log_loss = {log_loss_val.data}, train_acc = {train_accuracy:2f}%, validation_acc = {validation_accuracy:2f}%")

plot_series([("train", train_accuracies), ("validation", validation_accuracies)], "Accuracy")
losses_flat = [float(arr[0]) for arr in losses]
log_losses_flat = [float(arr[0]) for arr in log_losses]
# print(f"{losses=}")
# print(f"{log_losses=}")
# print(f"{validation_accuracies=}")
# print(f"{losses_flat=}")
# print(f"{log_losses_flat=}")
plot_series([("train", losses_flat), ("validation", log_losses_flat)], "Loss")
