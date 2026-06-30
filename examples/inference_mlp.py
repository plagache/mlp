from pathlib import Path

from dataset import compute_accuracy, create_data, load_dataset
from model_mlp import Network, load_json
from safetensors.numpy import load_file
from train_mlp import log_loss

from dlf.optimizer import get_parameters
from dlf.tensor import Tensor


def load_model(model: Network, input_file: str):
    state_dict = load_file(input_file)

    for i, layer in enumerate(model.layers):
        weight_key = f"l{i}.weight"
        bias_key = f"l{i}.bias"

        # also better expression or typing
        if weight_key in state_dict and bias_key in state_dict:
            saved_weight = state_dict[weight_key]
            saved_bias = state_dict[bias_key]
            assert saved_weight.shape == layer.weight.data.shape, f"Layer {i} weight shape mismatch: saved {saved_weight.shape} != model {layer.weight.data.shape}"
            assert saved_bias.shape == layer.bias.data.shape, f"Layer {i} bias shape mismatch: saved {saved_bias.shape} != model {layer.bias.data.shape}"
            layer.weight.data = saved_weight
            layer.bias.data = saved_bias
        else:
            print(f"Warning: No weights found for layer {i}")


if __name__ == "__main__":
    train_path, valid_path = create_data()
    X_test, Y_test = load_dataset(valid_path)

    assert Path("mlp.safetensors").exists(), "mlp.safetensors not found, run `uv run python examples/train_mlp.py` to generate it"
    assert Path("config.json").exists(), "config.json not found, you have to create it"

    layers_sizes = load_json("config.json")["layers"]
    model = Network(layers_sizes, X_test.shape[1])
    params = get_parameters(model)
    print(f"Optimizer is tracking {len(params)} parameters")
    # print(f"{params=}")
    load_model(model, "mlp.safetensors")
    # print(f"{params=}")

    Y_T = Tensor(Y_test)
    P_T = model(Tensor(X_test))
    validation_accuracy = compute_accuracy(Y_T.data, P_T.data)
    validation_loss = log_loss(Y_T, P_T)
    print(f"BCE: {validation_loss.data[-1]:4f}")
    print(f"Validation accuracy: {validation_accuracy:.2f}%")
