from enum import IntEnum, auto

import numpy as np


class Operations(IntEnum):
    ADD = auto()
    SUB = auto()
    SUM = auto()
    MUL = auto()
    DIV = auto()
    DOT = auto()
    RELU = auto()
    LOG = auto()
    EXP = auto()
    SOFTMAX = auto()
    T = auto()


backward_operations = {
    Operations.ADD: lambda gradient, parent: (gradient, gradient),
    Operations.SUB: lambda gradient, parent: (gradient, -gradient),
    Operations.SUM: lambda gradient, parent: (np.ones_like(parent[0].data) * gradient,),
    Operations.MUL: lambda gradient, parents: (parents[1].data * gradient, parents[0].data * gradient),
    Operations.DIV: lambda gradient, parents: (gradient / parents[1].data, (-parents[0].data * gradient / np.square(parents[1].data))),
    Operations.DOT: lambda gradient, parents: (gradient @ parents[1].data.T, parents[0].data.T @ gradient),
    Operations.RELU: lambda gradient, parent: (gradient * (np.where(parent[0].data <= 0, 0, 1)),),
    Operations.LOG: lambda gradient, parent: (gradient / parent[0].data,),
    Operations.EXP: lambda gradient, parent: (gradient * np.exp(parent[0].data),),
    Operations.SOFTMAX: lambda gradient, parent: (_softmax(parent[0].data) * (gradient - (gradient * _softmax(parent[0].data)).sum(axis=-1, keepdims=True)),),
    Operations.T: lambda gradient, parent: (gradient.T,),
}


# we are recomputing the softmax because we need the result in the lambda backwards
# this is highly inefficient
# a real deep learning framework is able to store the result and optimize its usage
# also separating Tensor operations from mathematical operations
def _softmax(x):
    """
    This function turns `raw guesses` into a clear probability distribution.

    1. Every number becomes positive.
    2. All the numbers together now add up to exactly 1.0.
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class Tensor:
    """
    4 + 5 = 9 <=> context: (`+`, 4, 5)
    2 * 3 = 6 <=> context: (`*`, 2, 3)
    """

    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)

        self.grad: np.ndarray | None = None
        self.context: tuple | None = None

    # a classmethod takes the class has its first input instead self
    @classmethod
    def _ensure_tensor(cls, x):
        return x if isinstance(x, cls) else cls(x)

    @classmethod
    def ones_like(cls, tensor):
        return cls(np.ones_like(tensor.data))

    def topo_sort(self):
        ret = dict()
        stack = [(self, False)]
        while stack:
            node, visited = stack.pop()
            if node in ret:
                continue
            if not visited:
                if node.context is not None:
                    stack.append((node, True))
                    ops, *parents = node.context
                    for parent in parents:
                        stack.append((parent, False))
            else:
                ret[node] = None
        return ret

    def backward(self):

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        for element in reversed(self.topo_sort()):
            ops, *parents = element.context
            backward_operation = backward_operations[ops]
            gradients = backward_operation(element.grad, [*parents])
            for parent, gradient in zip(parents, gradients):
                if parent.grad is None:
                    parent.grad = gradient
                else:
                    parent.grad += gradient
        return

    def __repr__(self):
        return f"<{self.data.shape}, {self.data}>"

    def __neg__(self):
        return self * -1

    def __getitem__(self, key):
        result = Tensor(self.data[key])
        return result

    def __add__(self, x):
        return self.ADD(x)

    def ADD(self, x):
        x = self._ensure_tensor(x)
        result = Tensor(self.data + x.data)
        result.context = (Operations.ADD, self, x)
        return result

    def __radd__(self, x):
        return self.ADD(x)

    def __sub__(self, x):
        return self.SUB(x)

    def SUB(self, x):
        x = self._ensure_tensor(x)
        result = Tensor(self.data - x.data)
        result.context = (Operations.SUB, self, x)
        return result

    def __rsub__(self, x):
        x = self._ensure_tensor(x)
        return x.SUB(self)

    def SUM(self):
        result = Tensor(np.sum(self.data))
        result.context = (Operations.SUM, self)
        return result

    def __mul__(self, x):
        return self.MUL(x)

    def MUL(self, x):
        x = self._ensure_tensor(x)
        result = Tensor(self.data * x.data)
        result.context = (Operations.MUL, self, x)
        return result

    def __rmul__(self, x):
        return self.MUL(x)

    def __truediv__(self, x):
        return self.DIV(x)

    def DIV(self, x):
        x = self._ensure_tensor(x)
        result = Tensor(self.data / x.data)
        result.context = (Operations.DIV, self, x)
        return result

    def __rtruediv__(self, x):
        x = self._ensure_tensor(x)
        return x.DIV(self)

    def __matmul__(self, x):
        return self.DOT(x)

    def MEAN(self):
        N = self.data.size
        fraction = Tensor([1 / N])
        return self.SUM().MUL(fraction)

    def DOT(self, x):
        """
        Matrix multiplication.

        Both operands must be 2D. For single samples, use shape (1, n) not (n,).
        Example: x = Tensor([[1.0, 2.0]])  # shape (1, 2), not Tensor([1.0, 2.0])
        No Broadcast
        1D @ 2D would require shape expand/ and reduce on specifique axis
        """
        result = Tensor(np.dot(self.data, x.data))
        result.context = (Operations.DOT, self, x)
        return result

    def RELU(self):
        """
        Activation function that silences negative values and keeps the positive ones.
        Mimicking a neuron, activating (firing) or not.
        Real neurons have a precise voltage, indicating precise values.
        """
        result = Tensor(np.where(self.data < 0, 0, self.data))
        result.context = (Operations.RELU, self)
        return result

    def log(self):
        return self.LOG()

    def LOG(self):
        result = Tensor(np.log(self.data))
        result.context = (Operations.LOG, self)
        return result

    def EXP(self):
        result = Tensor(np.exp(self.data))
        result.context = (Operations.EXP, self)
        return result

    def SOFTMAX(self):
        """
        Computes the softmax activation function along the last axis.
        Returns a Tensor of the same shape as the input.
        """
        result = Tensor(_softmax(self.data))
        result.context = (Operations.SOFTMAX, self)
        return result

    # Need to remove Transpose and perform it before creating the Tensor
    # Actually, it can be done on a numpy array that will feed to Tensor once the Transpose is done
    #
    # Or we would have to implement a Transpose Backward, that return the Transpose of the gradient
    # With @property we can then use y = x.T and it will call this method
    @property
    def T(self):
        result = type(self)(self.data.T)
        result.context = (Operations.T, self)
        return result
