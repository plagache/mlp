import numpy as np
from enum import auto, IntEnum


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
    SIGMOID = auto()
    T = auto()
    EXAMPLE = auto()

backward_operations = {
    Operations.ADD: lambda gradient, parent: (gradient, gradient),
    Operations.SUB: lambda gradient, parent: (gradient, -gradient),
    Operations.SUM: lambda gradient, parent: (np.ones_like(parent[0].data) * gradient,),
    Operations.MUL: lambda gradient, parents: (parents[1].data * gradient, parents[0].data * gradient),
    Operations.DIV: lambda gradient, parents: (gradient / parents[1].data, (-parents[0].data * gradient / np.square(parents[1].data))),
    Operations.DOT: lambda gradient, parents: (gradient @ parents[1].data.T, parents[0].data.T @ gradient),
    Operations.RELU: lambda gradient, parent: (gradient * (np.where(parent[0].data <= 0, 0, 1)),),
    Operations.LOG: lambda gradient, parent: (gradient * (1 / parent[0].data),),
    Operations.EXP: lambda gradient, parent: (gradient * np.exp(parent[0].data),),
    Operations.SOFTMAX: lambda gradient, parent: (None,),
    # np.exp(-parent[0].data) / (1 + np.exp(-parent[0].data)) = 1 - σ(x)
    # Proof: e^(-x)/(1+e^(-x)) = (1+e^(-x) - 1)/(1+e^(-x)) = 1 - 1/(1+e^(-x)) = 1 - σ(x)
    # σ′(x) = σ(x) · (1 - σ(x))
    # or σ′(x) = gradient * σ(x) * (1 - σ(x))
    # Operations.SIGMOID: lambda gradient, parent: (gradient * (1 / (1 + np.exp(-parent[0].data))) * (np.exp(-parent[0].data) / (1 + np.exp(-parent[0].data))),),
    Operations.T: lambda gradient, parent: (gradient.T,),
    Operations.EXAMPLE: lambda gradient, parent: (None,),
}


class Tensor:
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)

        self.grad: np.ndarray | None = None
        self.context: tuple | None = None

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
        """
        Input: a dict of nodes (result from topo_sort)
        apply backward with backward_operations[ops] on each nodes
        """

        # self.grad = np.array([1])
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

    def __add__(self, x):
        return self.ADD(x)

    def ADD(self, x):
        result = Tensor(self.data + x.data)
        result.context = (Operations.ADD, self, x)
        return result

    def __sub__(self, x):
        return self.SUB(x)

    def SUB(self, x):
        result = Tensor(self.data - x.data)
        result.context = (Operations.SUB, self, x)
        return result

    def SUM(self):
        result = Tensor(np.sum(self.data))
        result.context = (Operations.SUM, self)
        return result

    def __mul__(self, x):
        return self.MUL(x)

    def MUL(self, x):
        result = Tensor(self.data * x.data)
        result.context = (Operations.MUL, self, x)
        return result

    def __truediv__(self, x):
        return self.DIV(x)

    def DIV(self, x):
        result = Tensor(self.data / x.data)
        result.context = (Operations.DIV, self, x)
        return result

    def __matmul__(self, x):
        return self.DOT(x)

    def mean(self):
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
        result = Tensor(np.where(self.data < 0, 0, self.data))
        result.context = (Operations.RELU, self)
        return result

    def LOG(self):
        result = Tensor(np.log(self.data))
        result.context = (Operations.LOG, self)
        return result

    def EXP(self):
        result = Tensor(np.exp(self.data))
        result.context = (Operations.EXP, self)
        return result

    # input is a vector, of which we want to determine the highest probability
    def SOFTMAX(self):
        result = Tensor(np.exp(self.data) / np.sum(np.exp(self.data)))
        result.context = (Operations.SOFTMAX, self)
        return result

    def SIGMOID(self):
        result = Tensor(1 / (1 + np.exp(-self.data)))
        result.context = (Operations.SIGMOID, self)
        return result

    # Need to remove Transpose and perform it before creating the Tensor
    # Actually, it can be done on a numpy array that will feed to Tensor once the Transpose is done
    #
    # Or we would have to implement a Transpose Backward, that return the Transpose of the gradient
    @property
    def T(self):
        result = type(self)(self.data.T)
        result.context = (Operations.T, self)
        return result
