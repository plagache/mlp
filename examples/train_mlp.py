from dataset import compute_accuracy, create_data, load_dataset
from model_mlp import Network, get_parameters, get_state_dict, load_json, log_loss, save_model
from plot import plot_series

from dlf.optimizer import GD, Optimizer
from dlf.tensor import Tensor

def evaluate(model: Network, X, Y):
    prediction = model(Tensor(X))
    target = Tensor(Y)
    loss = log_loss(target, prediction)
    accuracy = compute_accuracy(target.data, prediction.data)
    return loss, accuracy

def train(model: Network, optimizer: Optimizer, X, Y, batch_size):
    for element in range(0, len(X), batch_size):
        X_batch = X[element : element + batch_size]
        Y_batch = Y[element : element + batch_size]

        Y_tensor = Tensor(Y_batch)
        Prediction_tensor = model(Tensor(X_batch))
        loss = log_loss(Y_tensor, Prediction_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    output_file = "mlp.safetensors"

    # Json
    layers_sizes = load_json("config.json")["layers"]
    seed = load_json("config.json")["seed"]

    train_path, valid_path = create_data(seed=seed)
    X_train, Y_train = load_dataset(train_path)
    print(f"X {train_path} shape: {X_train.shape}")
    X_validation, Y_validation = load_dataset(valid_path)
    print(f"X {valid_path} shape: {X_validation.shape}")

    model = Network(layers_sizes, X_train.shape[1])
    print(X_train.shape[0])
    print(X_train.shape[1])

    params = get_parameters(model)
    print(f"Optimizer is tracking {len(params)} parameters from {layers_sizes=}")
    # print(f"{params=}")

    weight_decay = 1e-7
    learning_rate = 0.001
    epochs = 100
    batch_size = 32

    optimizer = GD(get_parameters(model), learning_rate, weight_decay=weight_decay)
    # print(f"{params=}")

    validation_losses = []
    train_losses = []
    train_accuracies = []
    validation_accuracies = []

    for epoch in range(epochs):
        # for element in range(0, len(X_train), batch_size):
        #     X_batch = X_train[element : element + batch_size]
        #     Y_batch = Y_train[element : element + batch_size]
        #
        #     Y = Tensor(Y_batch)
        #     P = model(Tensor(X_batch))
        #     train_loss = log_loss(Y, P)
        #     optimizer.zero_grad()
        #     train_loss.backward()
        #     optimizer.step()
        #
        # Y_VAL = Tensor(Y_validation)
        # P_VAL = model(Tensor(X_validation))
        # validation_loss = log_loss(Y_VAL, P_VAL)
        # validation_losses.append(float(validation_loss.data[0]))
        #
        # train_losses.append(float(train_loss.data[0]))
        #
        # train_accuracy = compute_accuracy(Y.data, P.data)
        # validation_accuracy = compute_accuracy(Y_VAL.data, P_VAL.data)
        # train_accuracies.append(train_accuracy)
        # validation_accuracies.append(validation_accuracy)

        train(model, optimizer, X_train, Y_train, batch_size)
        train_loss, train_accuracy = evaluate(model, X_train, Y_train)
        train_losses.append(train_loss.data[0])


        print( f"Epoch {epoch+1}/{epochs} | "
               f"train loss: {train_losses[-1]:.4f} | "
               # f"validation loss: {validation_losses[-1]:.4f} | "
               f"train acc = {train_accuracy:.2f}% | "
               # f"validation acc = {validation_accuracy:.2f}%"
        )

    plot_series([("train", train_accuracies), ("validation", validation_accuracies)], "Accuracy")
    plot_series([("train", train_losses), ("validation", validation_losses)], "Loss")

    save_model(get_state_dict(model), output_file)
