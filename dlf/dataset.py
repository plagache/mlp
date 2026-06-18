from pathlib import Path

import numpy as np
import polars as pl


def decoder(inputs):
    """
    takes a [P, 1-P]
    and return the indices of the highest
    """
    output = np.argmax(inputs, axis=1)
    return output

def encoder(column):
    malign = column.replace({"M": 1, "B": 0}).cast(pl.Float64)
    benign = column.replace({"M": 0, "B": 1}).cast(pl.Float64)
    return np.stack([malign.to_numpy(), benign.to_numpy()], axis=1)

def compute_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    predictions_classes = decoder(predictions)
    targets_classes = decoder(targets)
    return np.mean(predictions_classes == targets_classes) * 100


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
    assert Path("data.csv").exists(), "data.csv not found"

    # Load and shuffle data
    data = pl.read_csv("data.csv", has_header=False)
    data = data.sample(fraction=1.0, shuffle=shuffle, seed=seed)

    # Detect bad data
    data_with_zero = data.filter(pl.any_horizontal(pl.selectors.numeric().eq(0)))
    print(f"Try columns with zero:\n{data_with_zero}")

    # Cleanup
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


def load_dataset(file_path):
    """
    do we store the normalisation technique ? yes in a safetensors
    """
    dataframe = pl.read_csv(file_path, has_header=False)
    # print(dataframe)

    # so the Malign is left and the Benign droite
    Y = encoder(dataframe["column_2"])
    X = dataframe.select(dataframe.columns[2:32]).to_numpy()

    # axis=0 so we have a mean for each features (30,) and not THE MEAN and a reduce axis ()
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    X_norm = (X - mean) / std

    print(f"X {file_path} shape: {X_norm.shape}")

    return X_norm, Y


if __name__ == "__main__":
    train_path, valid_path = create_data()
    X_train, Y_train = load_dataset(train_path)
    X_test, Y_test = load_dataset(valid_path)
