import os

from safetensors.numpy import load_file

from dlf.dataset import compute_accuracy, load_dataset
from dlf.model_mlp import Network
from dlf.tensor import Tensor

X_train, Y_train, X_test, Y_test = load_dataset()


model = Network()

assert os.path.exists("mlp.safetensors"), "mlp.safetensors not found, run `uv run python examples/train_mlp.py` to generate it"
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
