import os

import numpy as np
import polars as pl


def compute_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    predictions_classes = np.argmax(predictions, axis=1)
    targets_classes = np.argmax(targets, axis=1)
    return np.mean(predictions_classes == targets_classes) * 100


def load_dataset():
    assert os.path.exists("data.csv"), "data.csv not found"
    data = pl.read_csv("data.csv", has_header=False)
    print(data)

    data = data.with_columns(pl.col("column_2").replace({"M": 1, "B": 0}).cast(pl.Float64).alias("Malign"))
    data = data.with_columns(pl.col("column_2").replace({"M": 0, "B": 1}).cast(pl.Float64).alias("Benign"))

    Y = data.select(["Malign", "Benign"])
    X = data.select(data.columns[2:32])

    data_len = len(data)

    frac = int(0.8 * data_len)

    X_train = X[:frac].to_numpy()
    Y_train = Y[:frac].to_numpy()
    X_test = X[frac:].to_numpy()
    Y_test = Y[frac:].to_numpy()

    # axis=0 so we have a mean for each features (30,) and not THE MEAN and a reduce axis ()
    mean_train = X_train.mean(axis=0)
    std_train = X_train.std(axis=0)

    X_train_norm = (X_train - mean_train) / std_train
    X_test_norm = (X_test - mean_train) / std_train

    return X_train_norm, Y_train, X_test_norm, Y_test


def create_data():

    if os.path.exists("data_train.csv") and os.path.exists("data_valid.csv"):
        data_train = pl.read_csv("data_train.csv", has_header=False)
        data_valid = pl.read_csv("data_valid.csv", has_header=False)
        print(data_train)
        print(data_valid)
        print(f"data already split")
        return

    assert os.path.exists("data.csv"), "data.csv not found"
    data = pl.read_csv("data.csv", has_header=False)
    print(data)

    data_len = len(data)

    frac = int(0.8 * data_len)

    train = data[:frac]
    valid = data[frac:]
    train.write_csv("data_train.csv")
    valid.write_csv("data_valid.csv")
    # print(train)
    # print(valid)
    return


if __name__ == "__main__":
    # load_dataset()
    create_data()
