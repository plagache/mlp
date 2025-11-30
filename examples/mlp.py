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
        # self.weight1 = Tensor(np.random.uniform(-0.1, 0.1, (30,10)))
        # self.bias1 = Tensor(np.random.uniform(-0.1, 0.1, (10,)))
        # self.weight2 = Tensor(np.random.uniform(-0.1, 0.1, (10,30)))
        # self.bias2 = Tensor(np.random.uniform(-0.1, 0.1, (30,)))

    def __call__(self, x: Tensor):
        # x = x @ self.weight1
        # x = x + self.bias1
        # x = x @ self.weight2
        # x = x + self.bias2
        x = self.l1(x).RELU()
        x = self.l2(x).RELU()
        x = self.l3(x).RELU()
        x = self.l4(x).RELU()
        return x.SOFTMAX()

model = Network()

X_train, Y_train, X_test, Y_test = load_dataset()
# for the log_loss to work i need to rewrite the encoding of the Y_s
# create a tensor with one column per classes

steps = 5

# optimizer = SGD([model.weight1, model.bias1], 0.002)
# optimizer = SGD([model.l1.weight, model.l1.bias, model.l2.weight, model.l2.bias], 0.002)

def accuracy(true, prediction):
    true = np.argmax(true.to_numpy(), axis=-1)
    prediction = np.argmax(prediction.data, axis=-1)
    return (true == prediction).mean()

def log_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

for step in range(steps):
    print(step)

    Y_pred = model(Tensor(X_train))

    precision = accuracy(Y_train, Y_pred)
    print(f"{precision*100}%")

    loss = log_loss(Y_train.to_numpy(), Y_pred.data)
    print(f"{loss=}")

    # optimizer.step()
    # print(model.weight1)
