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
print(data.columns)

data = data.with_columns(
    pl.col("column_2").replace({"M": 1,"B": 0}).alias("column_2")
)
print(data)
types = data["column_2"].value_counts()
print(types)

class Network:
    def __init__(self):
        self.l1 = Tensor(np.random.rand(4,))
        self.l2 = Tensor(np.random.rand(4,))

    def __call__(self, x: Tensor):
        x = self.l1 * x
        x = self.l2 + x
        return x

model = Network()

input = Tensor([7, 9, 5, 3])

steps = 5

optimizer = SGD([model.l1, model.l2], 0.002)

for step in range(steps):
    print(step)
    optimizer.zero_grad()
    resultat = model(input)
    # loss = 
    resultat.backward()
    optimizer.step()
    print(resultat)
