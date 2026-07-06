import json
from pathlib import Path

from safetensors.numpy import load_file, save_file

from dlf.nn import Linear
from dlf.tensor import Tensor


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


def log_loss(y, p):
    return -((y * p.log() + (1 - y) * (1 - p).log()).MEAN())


def get_state_dict(model: Network) -> dict[str, Tensor]:
    state_dict = {}

    for i, layer in enumerate(model.layers):
        state_dict[f"layers.{i}.weight"] = layer.weight
        state_dict[f"layers.{i}.bias"] = layer.bias
    return state_dict


def get_parameters(model: Network) -> list[Tensor]:
    return list(get_state_dict(model).values())


# we can reduce here, but stay like this, to be sure i understand when i come back
def save_model(state_dict: dict, output_file: str):
    print(f"> saving model '{output_file}' to disk...")
    save_dict = {cle: valeur.data for cle, valeur in state_dict.items()}
    save_file(save_dict, output_file)


def load_model(model: Network, input_file: str) -> Network:
    state_dict = load_file(input_file)
    for i, layer in enumerate(model.layers):
        saved_weight = state_dict[f"layers.{i}.weight"]
        saved_bias = state_dict[f"layers.{i}.bias"]
        assert saved_weight.shape == layer.weight.data.shape, f"Layer {i} weight shape mismatch: saved {saved_weight.shape} != model {layer.weight.data.shape}"
        assert saved_bias.shape == layer.bias.data.shape, f"Layer {i} bias shape mismatch: saved {saved_bias.shape} != model {layer.bias.data.shape}"
        layer.weight.data = saved_weight
        layer.bias.data = saved_bias
    return model


def load_json(path: str) -> dict:
    with Path(path).open("r") as f:
        data = json.load(f)
        return data
