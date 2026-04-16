from safetensors.numpy import load_file
from dlf.tensor import Tensor
from dlf.nn import Linear
from dlf.dataset import load_dataset, compute_accuracy

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

loaded = load_file("mlp.safetensors")
model.l1.weight.data = loaded["l1.weight"]
model.l1.bias.data = loaded["l1.bias"]
model.l2.weight.data = loaded["l2.weight"]
model.l2.bias.data = loaded["l2.bias"]
model.l3.weight.data = loaded["l3.weight"]
model.l3.bias.data = loaded["l3.bias"]
model.l4.weight.data = loaded["l4.weight"]
model.l4.bias.data = loaded["l4.bias"]

Y_T = Tensor(Y_test)
P_T = model(Tensor(X_test))
validation_accuracy = compute_accuracy(Y_T.data, P_T.data)
print(f"Validation accuracy: {validation_accuracy:.2f}%")
