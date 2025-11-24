from tinygrad import Tensor, nn
from dlf.dataset import load_dataset
from tinygrad import Device
print(Device.DEFAULT)

class Network:
    def __init__(self):
        self.l1 = nn.Linear(30, 10)
        self.l2 = nn.Linear(10, 30)
    def __call__(self, x:Tensor) -> Tensor:
        x = self.l1(x)
        x = x.relu()
        x = self.l2(x)
        x = x.relu()
        return x.softmax()

model = Network()

X_train, Y_train, X_test, Y_test = load_dataset()

steps = 5
for step in range(steps):
    print(step)
    resultat = model(Tensor(X_train.to_numpy()))
    values = resultat.numpy()
    # print(resultat)
    print(values.tolist())
    # print(type(resultat))
