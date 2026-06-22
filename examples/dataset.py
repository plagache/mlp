from pathlib import Path

import numpy as np
import polars as pl
from safetensors.numpy import load_file, save_file


def create_data(percent=0.8, shuffle=True, seed=None):

    # we could add data_path has a parameters but what about already created train and valid then
    data_path = "data.csv"
    train_path = "data_train.csv"
    valid_path = "data_valid.csv"
    # no need for the function
    if Path(train_path).exists() and Path(valid_path).exists():
        print(f"data already split {train_path}, {valid_path}")
        return train_path, valid_path

    # cannot do the function
    assert Path(data_path).exists(), f"{data_path} not found"

    # Load and shuffle data
    data = pl.read_csv("data.csv", has_header=False)
    data = data.sample(fraction=1.0, shuffle=shuffle, seed=seed)

    # Detect bad data
    data_with_zero = data.filter(pl.any_horizontal(pl.selectors.numeric().eq(0)))
    print(f"Try columns with zero:\n{data_with_zero}")

    # Cleanup need to understand what ~ mean in this context
    data = data.filter(~pl.any_horizontal(pl.selectors.numeric().eq(0)))

    # print before split
    print(f"data being split:\n{data}")

    # Calcul du separateur
    data_len = len(data)
    frac = int(percent * data_len)

    # Split
    train = data[:frac]
    valid = data[frac:]

    # Write
    train.write_csv(train_path, include_header=False)
    valid.write_csv(valid_path, include_header=False)

    return train_path, valid_path


def decoder(predctions):
    """
    takes a [P, 1-P]
    and return the indices of the classe
    """
    classes = np.argmax(predctions, axis=1)
    return classes


def encoder(column):
    malign = column.replace({"M": 1, "B": 0}).cast(pl.Float64)
    benign = column.replace({"M": 0, "B": 1}).cast(pl.Float64)
    return np.stack([malign.to_numpy(), benign.to_numpy()], axis=1)


def compute_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    predictions_classes = decoder(predictions)
    targets_classes = decoder(targets)
    return np.mean(predictions_classes == targets_classes) * 100


def normalisation(X, file_path):
    stats_path = "norm_stats.safetensors"

    if Path(stats_path).exists():
        print(f"Stats already compute loading from: {stats_path}")
        stats = load_file(stats_path)
        mean = stats["mean"]
        std = stats["std"]
    else:
        # axis=0 so we have a mean for each features (30,) and not THE MEAN and a reduce axis ()
        mean = X.mean(axis=0)
        std = X.std(axis=0)

        save_file({"mean": mean, "std": std}, stats_path)
        print(f"> saving stats '{stats_path}' to disk...")

    X_norm = (X - mean) / std

    return X_norm


def load_dataset(file_path):
    dataframe = pl.read_csv(file_path, has_header=False)

    Y = encoder(dataframe["column_2"])

    X = dataframe.select(dataframe.columns[2:32]).to_numpy()
    X_norm = normalisation(X, file_path)

    return X_norm, Y


if __name__ == "__main__":
    train_path, valid_path = create_data()
    X_train, Y_train = load_dataset(train_path)
    print(f"X {train_path} shape: {X_train.shape}")
    X_test, Y_test = load_dataset(valid_path)
    print(f"X {valid_path} shape: {X_test.shape}")
