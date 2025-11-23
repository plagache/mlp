import numpy as np
import random
from dlf.tensor import Tensor
from dlf.optimizer import SGD
from dlf.dataset import load_dataset

class Network:
    def __init__(self):
        self.weight1 = Tensor(np.random.rand(30,))
        self.bias1 = Tensor(np.random.rand(30,))
        self.weight2 = Tensor(np.random.rand(30,))
        self.bias2 = Tensor(np.random.rand(30,))

    def __call__(self, x: Tensor):
        x = self.weight1 * x
        x = self.bias1 + x
        x = x.RELU()
        x = self.weight2 * x
        x = self.bias2 + x
        x = x.RELU()
        return x.SOFTMAX()

model = Network()

X_train, Y_train, X_test, Y_test = load_dataset()

steps = 5

optimizer = SGD([model.weight1, model.bias1], 0.002)

for step in range(steps):
    print(step)
    optimizer.zero_grad()
    # print(Tensor(X_train).data)
    # print(X_train)
    resultat = model(Tensor(X_train))
    # loss = 
    # resultat.backward()
    # optimizer.step()
    # print(model.weight1)
    print(resultat)
    print(type(resultat))
