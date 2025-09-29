import numpy as np
from dlf.tensor import Tensor
from dlf.optimizer import SGD, get_parameters
import math

class Network:
    def __init__(self):
        self.l1 = Tensor(np.random.rand(4,))
        self.l2 = Tensor(np.random.rand(4,))
        print(self.l1)

    def __call__(self, x: Tensor):
        x = self.l1 * x
        x = self.l2 + x
        return x

model = Network()

params = get_parameters(model)
# print("parameters: ", params)

input = Tensor([7, 9, 5, 3])

steps = 5

optimizer = SGD(list(params.values()), 0.002)

for step in range(steps):
    print(step)
    resultat = model(input).SUM()
    resultat.backward()
    optimizer.step()
