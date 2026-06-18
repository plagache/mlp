from dlf.tensor import Tensor
from dlf.nn import Linear


class Network:
    def __init__(self):
        self.l1 = Linear(30, 30)
        self.l2 = Linear(30, 30)
        self.l3 = Linear(30, 10)
        self.l4 = Linear(10, 2)

    def __call__(self, x: Tensor):
        x = self.l1(x).RELU()
        x = self.l2(x).RELU()
        x = self.l3(x).RELU()
        x = self.l4(x).SOFTMAX()
        return x
