import os
# import Path

import numpy as np
import polars as pl


def compute_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    predictions_classes = np.argmax(predictions, axis=1)
    targets_classes = np.argmax(targets, axis=1)
    return np.mean(predictions_classes == targets_classes) * 100


def create_data(percent=0.8, shuffle=True, seed=None):

    # no need for the function
    if os.path.exists("data_train.csv") and os.path.exists("data_valid.csv"):
        data_train = pl.read_csv("data_train.csv", has_header=False)
        data_valid = pl.read_csv("data_valid.csv", has_header=False)
        print(f"data already split")
        return

    # cannot do the function
    assert os.path.exists("data.csv"), "data.csv not found"
    data = pl.read_csv("data.csv", has_header=False)
    data.sample(fraction=percent, shuffle=shuffle, seed=seed)
    print(f"data being split:\n{data}")

    data_len = len(data)

    frac = int(percent * data_len)

    train = data[:frac]
    valid = data[frac:]
    train.write_csv("data_train.csv", include_header=False)
    valid.write_csv("data_valid.csv", include_header=False)
    return


def load_dataset():
    if not os.path.exists("data_train.csv") or os.path.exists("data_valid.csv"):
        create_data()
    data_train = pl.read_csv("data_train.csv", has_header=False)
    data_valid = pl.read_csv("data_valid.csv", has_header=False)
    # print(data_train)
    # print(data_valid)

    data_train = data_train.with_columns(pl.col("column_2").replace({"M": 1, "B": 0}).cast(pl.Float64).alias("Malign"))
    data_train = data_train.with_columns(pl.col("column_2").replace({"M": 0, "B": 1}).cast(pl.Float64).alias("Benign"))
    data_valid = data_valid.with_columns(pl.col("column_2").replace({"M": 1, "B": 0}).cast(pl.Float64).alias("Malign"))
    data_valid = data_valid.with_columns(pl.col("column_2").replace({"M": 0, "B": 1}).cast(pl.Float64).alias("Benign"))

    Y_train = data_train.select(["Malign", "Benign"]).to_numpy()
    X_train = data_train.select(data_train.columns[2:32]).to_numpy()
    Y_test = data_valid.select(["Malign", "Benign"]).to_numpy()
    X_test = data_valid.select(data_valid.columns[2:32]).to_numpy()

    # axis=0 so we have a mean for each features (30,) and not THE MEAN and a reduce axis ()
    mean_train = X_train.mean(axis=0)
    std_train = X_train.std(axis=0)

    X_train_norm = (X_train - mean_train) / std_train
    X_test_norm = (X_test - mean_train) / std_train

    return X_train_norm, Y_train, X_test_norm, Y_test


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_dataset()
    print(f"{X_train.shape=}\n{Y_train.shape=}\n{X_test.shape=}\n{Y_test.shape=}")
