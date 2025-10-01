import numpy as np
from dlf.tensor import Tensor
from dlf.optimizer import SGD

class Network:
    def __init__(self):
        self.l1 = Tensor(np.random.rand(4,))
        self.l2 = Tensor(np.random.rand(4,))

    def __call__(self, x: Tensor):
        x = self.l1 * x
        x = self.l2 + x
        return x

model = Network()

input = Tensor([7, 9, 5, 3])

steps = 5

optimizer = SGD([model.l1, model.l2], 0.002)

for step in range(steps):
    print(step)
    optimizer.zero_grad()
    resultat = model(input).SUM()
    resultat.backward()
    optimizer.step()
