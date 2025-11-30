import polars as pl

def load_dataset():
    # there is no missing value
    # we should look into regularisation
    data = pl.read_csv("data.csv", has_header=False)

    data = data.with_columns(pl.col("column_2").replace({"M": 1,"B": 0}).cast(pl.Float64).alias("Malign"))
    data = data.with_columns(pl.col("column_2").replace({"M": 0,"B": 1}).cast(pl.Float64).alias("Benign"))

    Y = data["Malign", "Benign"]
    X = data.select(data.columns[2:32])

    data_len = len(data)

    frac = int(0.8 * data_len)

    return X[:frac], Y[:frac], X[frac:], Y[frac:]
