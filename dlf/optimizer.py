from .tensor import Tensor


def get_parameters(model):
    parameters = []
    for attr in vars(model).values():
        if isinstance(attr, Tensor):
            parameters.append(attr)
            print(attr)
        elif hasattr(attr, "__dict__"):
            parameters.extend(get_parameters(attr))
    return parameters


class GD:
    def __init__(self, parameters: list[Tensor], learning_rate: float):
        self.params = parameters
        self.lr = learning_rate

    def step(self):
        for param in self.params:
            if param.grad is not None:
                if param.grad.shape != param.data.shape:
                    param.grad = param.grad.sum(axis=0)
                param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            param.grad = None
