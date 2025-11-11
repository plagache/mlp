import numpy as np
from enum import auto, IntEnum

class Operations(IntEnum):
    ADD = auto()
    MUL = auto()
    SUM = auto()
    NEG = auto()

# lets carefully check that we are computing, with same type
backward_operations = {
    Operations.ADD: lambda gradient, parent: (gradient, gradient),
    Operations.MUL: lambda gradient, parents: (parents[1].data * gradient, parents[0].data * gradient),
    Operations.SUM: lambda gradient, parent: (np.full_like(parent, gradient)),
    Operations.NEG: lambda gradient, parents: (None),
}

class Tensor():
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)

        self.grad: np.ndarray = None
        self.context = None

    def topo_sort(self):
        ret = dict()
        stack = [(self, False)]
        while stack:
            node, visited = stack.pop()
            if node in ret: continue
            if not visited:
                if node.context is not None:
                    stack.append((node, True))
                    ops, *parents = node.context
                    for parent in parents: stack.append((parent, False))
            else:
                ret[node] = None
        return ret


    def backward(self):
        """
        input: a list of nodes as a paramater (the result from the topo_sort)
        apply backward from backward_operations[ops] on each nodes
        """

        operations = []
        # self.grad = np.array(1)
        self.grad = np.ones_like(self.data)


        for element in reversed(self.topo_sort()):
            ops, *parents = element.context
            backward_operation = backward_operations[ops]
            gradients = backward_operation(element.data, [*parents])
            for parent, gradient in zip(parents, gradients):
                if parent.grad is None:
                    parent.grad = gradient
                else:
                    parent.grad += gradient

        list_ops = []
        for operation in operations:
            tensor, ops = operation
            list_ops.append(f"{ops} : {tensor.data.shape}")

        result = " ---> ".join(list_ops)
        print(f"\n\n{result}")

        return

    def __repr__(self):
        if self.context is not None:
            # print(self.context)
            ops, *parents = self.context
            return f"<{self.data.shape}, {self.data}, {ops}>"
        else:
            return f"<{self.data.shape}, {self.data}>"

    def __add__(self, x):
        return self.ADD(x)

    def ADD(self, x):
        result = Tensor(self.data + x.data)
        result.context = (Operations.ADD, self, x)
        return result

    def __mul__(self, x):
        return self.MUL(x)

    def MUL(self, x):
        result = Tensor(self.data * x.data)
        result.context = (Operations.MUL, self, x)
        return result

    def SUM(self):
        result = Tensor(np.sum(self.data))
        result.context = (Operations.SUM, self)
        return result
