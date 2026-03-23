import math
import numpy as np
from .tensor import Tensor


class Linear:
    """
    Linear (fully-connected) layer.

    Input x must be 2D with shape (batch_size, in_features).
    For single samples, use shape (1, in_features) not (in_features,).
    """

    def __init__(self, in_features, out_features):
        """
        use the he uniform to better work with RELU
        """
        bound = np.sqrt(6/in_features)
        self.weight = Tensor(np.random.uniform(-bound, bound, (in_features, out_features)))
        self.bias = Tensor(np.zeros((out_features,)))

    def __call__(self, x):
        return x @ self.weight + self.bias
