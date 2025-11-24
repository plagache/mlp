import numpy as np
import random
from dlf.tensor import Tensor
from dlf.optimizer import SGD
from dlf.dataset import load_dataset
from dlf.nn import Linear

class Network:
    def __init__(self):
        self.l1 = Linear(30, 10)
        self.l2 = Linear(10, 30)
        # self.weight1 = Tensor(np.random.uniform(-0.1, 0.1, (30,10)))
        # self.bias1 = Tensor(np.random.uniform(-0.1, 0.1, (10,)))
        # self.weight2 = Tensor(np.random.uniform(-0.1, 0.1, (10,30)))
        # self.bias2 = Tensor(np.random.uniform(-0.1, 0.1, (30,)))

    def __call__(self, x: Tensor):
        # x = x @ self.weight1
        # x = x + self.bias1
        x = self.l1(x)
        x = x.RELU()
        x = self.l2(x)
        # x = x @ self.weight2
        # x = x + self.bias2
        x = x.RELU()
        return x.SOFTMAX()

model = Network()

X_train, Y_train, X_test, Y_test = load_dataset()

steps = 5

# optimizer = SGD([model.weight1, model.bias1], 0.002)
optimizer = SGD([model.l1.weight, model.l1.bias, model.l2.weight, model.l2.bias], 0.002)

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
