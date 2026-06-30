import json
from pathlib import Path

from dlf.nn import Linear
from dlf.tensor import Tensor


def load_json(path: str) -> dict:
    with Path(path).open("r") as f:
        data = json.load(f)
        return data


class Network:
    def __init__(self, layers_sizes: list[int], input_dim: int):
        self.layers = []
        for output_dim in layers_sizes:
            self.layers.append(Linear(input_dim, output_dim))
            input_dim = output_dim

    def __call__(self, x: Tensor):
        for layer in self.layers[:-1]:
            x = layer(x).RELU()
        x = self.layers[-1](x).SOFTMAX()
        return x
