from dataset import compute_accuracy, create_data, load_dataset
from model_mlp import Network
from safetensors.numpy import save_file

from dlf.optimizer import GD, get_parameters
from dlf.plot import plot_series
from dlf.tensor import Tensor


def log_loss(y, p):
    return -((y * p.log() + (1 - y) * (1 - p).log()).MEAN())


def save_model(model: Network, output_file: str):
    state_dict = {}

    for i, layer in enumerate(model.layers):
        print(f"{i}, {layer}")
        state_dict[f"l{i}.weight"] = layer.weight.data
        state_dict[f"l{i}.bias"] = layer.bias.data

    save_file(state_dict, output_file)
    print(f"> saving modular model '{output_file}' to disk...")


if __name__ == "__main__":
    output_file = "mlp.safetensors"

    train_path, valid_path = create_data()
    X_train, Y_train = load_dataset(train_path)
    print(f"X {train_path} shape: {X_train.shape}")
    X_test, Y_test = load_dataset(valid_path)
    print(f"X {valid_path} shape: {X_test.shape}")

    layer_sizes = [30, 30, 10, 2]
    model = Network(layer_sizes, X_train.shape[1])
    print(X_train.shape[0])
    print(X_train.shape[1])

    params = get_parameters(model)
    print(f"Optimizer is tracking {len(params)} parameters from {layer_sizes=}")
    # print(f"{params=}")

    # optimizer = GD(get_parameters(model), 0.002, weight_decay=0)
    optimizer = GD(get_parameters(model), 0.001, weight_decay=1e-7)
    # print(f"{params=}")

    validation_losses = []
    train_losses = []
    train_accuracies = []
    validation_accuracies = []

    # steps = 2000
    # for step in range(steps):
    epochs = 100
    batch_size = 32
    for epoch in range(epochs):
        for e in range(0, len(X_train), batch_size):
            X_batch = X_train[e : e + batch_size]
            Y_batch = Y_train[e : e + batch_size]
            Y = Tensor(Y_batch)
            P = model(Tensor(X_batch))

            train_loss = log_loss(Y, P)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # if (step + 1) % 10 == 0:
        Y_T = Tensor(Y_test)
        P_T = model(Tensor(X_test))

        validation_loss = log_loss(Y_T, P_T)

        validation_losses.append(float(validation_loss.data[0]))
        train_losses.append(float(train_loss.data[0]))

        train_accuracy = compute_accuracy(Y.data, P.data)
        validation_accuracy = compute_accuracy(Y_T.data, P_T.data)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        print(f"epoch {epoch}/{epochs} - loss: {train_losses[-1]:.4f}, Accuracy = {train_accuracy:.2f}%")
        # print(f"epoch {epoch}/{epochs} - loss: {train_losses[-1]:.4f}, validation_loss: {validation_losses[-1]:.4f} - train_acc = {train_accuracy:.2f}%, validation_acc = {validation_accuracy:.2f}%")
        # print(f"step {step + 1 % 10}/{steps} - loss: {train_losses[-1]:.4f}, validation_loss: {validation_losses[-1]:.4f} - train_acc = {train_accuracy:.2f}%, validation_acc = {validation_accuracy:.2f}%")

    plot_series([("train", train_accuracies), ("validation", validation_accuracies)], "Accuracy")
    plot_series([("train", train_losses), ("validation", validation_losses)], "Loss")

    save_model(model, output_file)
