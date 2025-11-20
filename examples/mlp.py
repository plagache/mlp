import numpy as np
import random
import polars as pl
from dlf.tensor import Tensor
from dlf.optimizer import SGD

random.seed(42)

# there is no missing value
# we should look into regularisation
# Transform M-B in 0-1
data = pl.read_csv("data.csv", has_header=False)
# print(data.columns)

data = data.with_columns(
    pl.col("column_2").replace({"M": 1,"B": 0}).cast(pl.Float64).alias("column_2")
)
# print(data)
# types = data["column_2"].value_counts()
# print(types)

Y = Tensor(data["column_2"])
X = data.select(data.columns[2:])
data_len = len(data)
print(data_len)
frac = int(0.8 * data_len)
X_train = X[:frac]
X_test = X[frac:]
print(X_train)
print(X_test)
exit() 

class Network:
    def __init__(self):
        self.weight1 = Tensor(np.random.rand(30,))
        self.bias1 = Tensor(np.random.rand(30,))

    def __call__(self, x: Tensor):
        x = self.weight1 * x
        x = self.bias1 + x
        return x.SIGMOID()

model = Network()

target = Tensor(data["column_2"])
inputs = Tensor(data.select(data.columns[2:]))

steps = 5

optimizer = SGD([model.weight1, model.bias1], 0.002)

for step in range(steps):
    print(step)
    optimizer.zero_grad()
    resultat = model(inputs)
    # loss = 
    resultat.backward()
    optimizer.step()
    print(model.weight1)
    print(resultat)
