import numpy as np
from tinygrad import Tensor, nn
from dlf.dataset import load_dataset
from tinygrad import Device
print(Device.DEFAULT)

class Network:
    def __init__(self):
        self.l1 = nn.Linear(30, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, 10)
        self.l4 = nn.Linear(10, 2)
    def __call__(self, x:Tensor) -> Tensor:
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        x = self.l3(x).relu()
        x = self.l4(x).relu()
        return x.softmax()

model = Network()

X_train, Y_train, X_test, Y_test = load_dataset()

steps = 5
for step in range(steps):
    print(step)
    resultat = model(Tensor(X_train.to_numpy()))
    values = resultat.numpy()
    probability = np.argmax(values, -1)
    # print(resultat)
    print(values.tolist())
    print(f"{probability=}")
    # print(type(resultat))
