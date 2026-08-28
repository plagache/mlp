from dataset import compute_accuracy, create_data, load_dataset
from model_mlp import Network, get_parameters, get_state_dict, load_json, log_loss, save_model
from plot import plot_series

from dlf.optimizer import GD, Optimizer
from dlf.tensor import Tensor


def evaluate(model: Network, X: Tensor, Y: Tensor) -> tuple[float, float]:
    prediction = model(X)
    loss = log_loss(Y, prediction)
    accuracy = compute_accuracy(Y.data, prediction.data)
    return loss.data.item(), accuracy


def train(model: Network, optimizer: Optimizer, X: Tensor, Y: Tensor, batch_size: int):
    for element in range(0, len(X.data), batch_size):
        X_batch = X[element : element + batch_size]
        Y_batch = Y[element : element + batch_size]
        Prediction_tensor = model(X_batch)
        loss = log_loss(Y_batch, Prediction_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def fit(model, optimizer, data, epochs, batch_size):
    X_train, Y_train = data["train"]
    X_validation, Y_validation = data["validation"]
    metrics = {"validation_loss": [], "train_loss": [], "train_accuracy": [], "validation_accuracy": []}

    for epoch in range(epochs):
        train(model, optimizer, X_train, Y_train, batch_size)
        train_loss, train_accuracy = evaluate(model, X_train, Y_train)
        metrics["train_loss"].append(train_loss)
        metrics["train_accuracy"].append(train_accuracy)

        validation_loss, validation_accuracy = evaluate(model, X_validation, Y_validation)
        metrics["validation_loss"].append(validation_loss)
        metrics["validation_accuracy"].append(validation_accuracy)

        print(f"Epoch {epoch + 1}/{epochs} | train loss: {train_loss:.4f} | validation loss: {validation_loss:.4f} | train acc = {train_accuracy:.2f}% | validation acc = {validation_accuracy:.2f}%")
    return metrics


if __name__ == "__main__":
    output_file = "mlp.safetensors"

    # Json
    layers_sizes = load_json("config.json")["layers"]
    seed = load_json("config.json")["seed"]

    train_path, valid_path = create_data(seed=seed)
    X_train, Y_train, X_train_Shape, Y_train_Shape = load_dataset(train_path)
    X_validation, Y_validation, _, _ = load_dataset(valid_path)
    print(f"X {train_path} shape: {X_train_Shape}, {Y_train_Shape}")
    # print(f"X {valid_path} shape: {X_validation_Shape}, {Y_validation_Shape}")

    data = {"train": (Tensor(X_train), Tensor(Y_train)), "validation": (Tensor(X_validation), Tensor(Y_validation))}

    # we can put them in json and initialize up there
    weight_decay = 1e-7
    learning_rate = 0.001
    epochs = 100
    # here caution with small batch_size resulting in log of 0 or divided by zero
    batch_size = 32

    model = Network(layers_sizes, X_train_Shape[1])
    # print(X_train_Shape[0])
    # print(X_train_Shape[1])

    params = get_parameters(model)
    print(f"Optimizer is tracking {len(params)} parameters from {layers_sizes=}")

    optimizer = GD(get_parameters(model), learning_rate, weight_decay=weight_decay)

    metrics = fit(model, optimizer, data, epochs, batch_size)

    plot_series([("train", metrics["train_accuracy"]), ("validation", metrics["validation_accuracy"])], "Accuracy")
    plot_series([("train", metrics["train_loss"]), ("validation", metrics["validation_loss"])], "Loss")

    save_model(get_state_dict(model), output_file)
