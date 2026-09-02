from pathlib import Path

from dataset import compute_accuracy, create_data, load_dataset
from model_mlp import Network, load_json, load_model
from train_mlp import evaluate

from dlf.tensor import Tensor

if __name__ == "__main__":
    train_path, valid_path = create_data()
    X_validation, Y_validation, X_validation_Shape, Y_validation_Shape = load_dataset(valid_path)

    assert Path("mlp.safetensors").exists(), "mlp.safetensors not found, run `uv run python examples/train_mlp.py` to generate it"
    assert Path("config.json").exists(), "config.json not found, you have to create it"

    # Reconstruct the model arch
    layers_sizes = load_json("config.json")["layers"]
    model = Network(layers_sizes, X_validation_Shape[1])

    # Load parameters
    load_model(model, "mlp.safetensors")

    # Evaluate
    validation_loss, validation_accuracy = evaluate(model, Tensor(X_validation), Tensor(Y_validation))
    print(f"BCE: {validation_loss:4f}")
    print(f"Validation accuracy: {validation_accuracy:.2f}%")
