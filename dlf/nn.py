import math
import numpy as np
from .tensor import Tensor


class Linear:
    def __init__(self, in_features, out_features):
        bound = 1 / math.sqrt(in_features)
        self.weight = Tensor(np.random.uniform(-bound, bound, (in_features, out_features)))
        self.bias = Tensor(np.random.uniform(-bound, bound, (out_features,)))

    def __call__(self, x):
        return x @ self.weight + self.bias
