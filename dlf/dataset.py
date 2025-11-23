import polars as pl

def load_dataset():
    # there is no missing value
    # we should look into regularisation
    # Transform M-B in 0-1
    data = pl.read_csv("data.csv", has_header=False)
    # print(data.columns)

    data = data.with_columns(pl.col("column_2").replace({"M": 1,"B": 0}).cast(pl.Float64).alias("column_2"))

    Y = data["column_2"]
    X = data.select(data.columns[2:])

    data_len = len(data)

    frac = int(0.8 * data_len)

    return X[:frac], Y[:frac], X[frac:], Y[frac:]

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_dataset()
    print("X_train :", X_train)
