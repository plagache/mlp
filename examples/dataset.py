from pathlib import Path

import numpy as np
import polars as pl
from safetensors.numpy import load_file, save_file

# maybe dataset.py -> data_pipeline.py
# create_data(): test if data exist / load_csv / cleanup data / call split_data -> return path to created data


def create_data(percent=0.8, shuffle=True, seed=None) -> tuple[str | Path, str | Path]:
    """
    we should type the return
    probably rename split_data
    """

    # we could add data_path has a parameters but what about already created train and valid then
    data_path = "data.csv"
    train_path = "data_train.csv"
    valid_path = "data_valid.csv"
    # no need to create_data
    if Path(train_path).exists() and Path(valid_path).exists():
        print(f"data already split {train_path}, {valid_path}")
        return train_path, valid_path

    # cannot do create_data
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
    print(seed)

    return train_path, valid_path


def decoder(predictions) -> np.ndarray:
    """
    takes a [P, 1-P]
    and return the indices of the classe
    """
    classes = np.argmax(predictions, axis=1)
    return classes


def encoder(column):
    """
    should take a dataframe ? not sure
    but we can type the return
    """
    malign = column.replace({"M": 1, "B": 0}).cast(pl.Float64)
    benign = column.replace({"M": 0, "B": 1}).cast(pl.Float64)
    return np.stack([malign.to_numpy(), benign.to_numpy()], axis=1)


def compute_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    predictions_classes = decoder(predictions)
    targets_classes = decoder(targets)
    return np.mean(predictions_classes == targets_classes) * 100


stats_path = Path("norm_stats.safetensors")


def normalisation(X: np.ndarray, path: str | Path = stats_path) -> np.ndarray:
    if Path(path).exists():
        mean, std = load_normalisation(path)
    else:
        mean, std = fit_normalisation(X)
        save_normalisation(mean, std, path)
    return transform(X, mean, std)


def load_normalisation(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    stats = load_file(path)
    return stats["mean"], stats["std"]


def save_normalisation(mean: np.ndarray, std: np.ndarray, path: str | Path):
    save_file({"mean": mean, "std": std}, path)
    # how to print / log using environement variable for ex:DEBUG ?
    # print(f"> saving normalisation stats '{path}' to disk...")


def fit_normalisation(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # axis=0 so we have a mean for each features (30,) and not THE MEAN and a reduce axis ()
    return X.mean(axis=0), X.std(axis=0)


def transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def load_csv(path: str | Path) -> pl.DataFrame:
    """
    https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names
    """
    return pl.read_csv(path, has_header=False)


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple, tuple]:
    dataframe = load_csv(path)

    # should be a const
    Y = encoder(dataframe["column_2"])

    X = dataframe.select(dataframe.columns[2:]).to_numpy()
    X_norm = normalisation(X, stats_path)

    return X_norm, Y, X_norm.shape, Y.shape


if __name__ == "__main__":
    train_path, valid_path = create_data()
    X_train, Y_train, X_train_Shape, Y_train_Shape = load_dataset(train_path)
    print(f"X {train_path} shape: {X_train_Shape}, {Y_train_Shape}")
    X_validation, Y_validation, X_validation_Shape, Y_validation_Shape = load_dataset(valid_path)
    print(f"X {valid_path} shape: {X_validation_Shape}, {Y_validation_Shape}")
