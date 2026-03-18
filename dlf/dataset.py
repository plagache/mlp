import polars as pl

def load_dataset():
    """
    there is no missing value
    we should look into regularisation / normalization

    need mean and variance calc for all features
    """
    data = pl.read_csv("data.csv", has_header=False)

    data = data.with_columns(pl.col("column_2").replace({"M": 1,"B": 0}).cast(pl.Float64).alias("Malign"))
    data = data.with_columns(pl.col("column_2").replace({"M": 0,"B": 1}).cast(pl.Float64).alias("Benign"))

    Y = data.select(["Malign", "Benign"])
    X = data.select(data.columns[2:32])

    data_len = len(data)

    frac = int(0.8 * data_len)

    X_train = X[:frac]
    Y_train = Y[:frac]
    X_test = X[frac:]
    Y_test = Y[frac:]

    mean_train = X_train.mean()
    std_train = X_train.std()

    X_train_norm = (X_train - mean_train) / std_train
    X_test_norm = (X_test - mean_train) / std_train

    return X_train_norm, Y_train, X_test_norm, Y_test
