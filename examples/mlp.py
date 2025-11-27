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

def accuracy(prediction, true):
    return (prediction == true.to_numpy()).mean()

def log_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

for step in range(steps):
    print(step)
    # optimizer.zero_grad()
    # print(Tensor(X_train).data)
    # print(X_train)
    Y_pred = model(Tensor(X_train))
    resultat = np.argmax(Y_pred.data, -1)
    precision = accuracy(resultat, Y_train)
    # accuracy = (Y_pred == Tensor(Y_train).data).mean()
    # loss = log_loss(Y_train, Y_pred.data)
    print(f"{(Y_train * np.log(Y_pred.data) + (1 - Y_train) * np.log(1 - Y_pred.data))=}")
    print(f"{Y_train}")
    # print(f"{loss=}")
    # optimizer.step()
    # print(model.weight1)
    # print(f"{Y_pred=}")
    # print(y_train * np.log(y_pred))
    # print(f"{y_train.to_numpy()=}")
    # print(Y_pred == Y_train.to_numpy())
    # print(f"{precision*100}%")
