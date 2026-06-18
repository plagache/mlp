import random
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from dlf.dataset import compute_accuracy, load_dataset, create_data
from dlf.model_mlp import Network
from dlf.nn import Linear
from dlf.optimizer import GD, get_parameters
from dlf.plot import plot_series
from dlf.tensor import Tensor


def log_loss(y, p):
    return -((y * p.log() + (1 - y) * (1 - p).log()).MEAN())


if __name__ == "__main__":
    output_file = "mlp.safetensors"

    train_path, valid_path = create_data()
    X_train, Y_train = load_dataset(train_path)
    X_test, Y_test = load_dataset(valid_path)

    model = Network()

    # optimizer = GD(get_parameters(model), 0.002, weight_decay=0)
    optimizer = GD(get_parameters(model), 0.001, weight_decay=1e-7)

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

    save_file(
        {
            "l1.weight": model.l1.weight.data,
            "l1.bias": model.l1.bias.data,
            "l2.weight": model.l2.weight.data,
            "l2.bias": model.l2.bias.data,
            "l3.weight": model.l3.weight.data,
            "l3.bias": model.l3.bias.data,
            "l4.weight": model.l4.weight.data,
            "l4.bias": model.l4.bias.data,
        },
        output_file,
    )
    print(f"> saving model '{output_file}' to disk...")
