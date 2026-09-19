from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

stats_path = Path("norm_stats.safetensors")


def normalisation(X: np.ndarray, path: str | Path = Path("norm_stats.safetensors")) -> np.ndarray:
    if Path(path).exists():
        mean, std = load_normalisation(path)
    else:
        mean, std = fit_normalisation(X)
        save_normalisation(mean, std, path)
    return transform(X, mean, std)


def load_normalisation(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    stats = load_file(path)
    return stats["mean"], stats["std"]


def save_normalisation(mean: np.ndarray, std: np.ndarray, path: str | Path) -> None:
    save_file({"mean": mean, "std": std}, path)
    # how to print / log using environement variable for ex:DEBUG ?
    # print(f"> saving normalisation stats '{path}' to disk...")


def fit_normalisation(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # axis=0 so we have a mean for each features (30,) and not THE MEAN and a reduce axis ()
    return X.mean(axis=0), X.std(axis=0)


def transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std
