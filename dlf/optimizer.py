from .tensor import Tensor


def get_parameters(model):
    parameters = []
    for object in vars(model).values():
        if isinstance(object, Tensor):
            parameters.append(object)
        elif hasattr(object, "__dict__"):
            parameters.extend(get_parameters(object))
    return parameters


class GD:
    def __init__(self, parameters: list[Tensor], learning_rate: float, weight_decay=0.0):
        self.params = parameters
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            if param.grad is not None:
                if param.grad.shape != param.data.shape:
                    param.grad = param.grad.sum(axis=0)
                # param.data -= self.lr * param.grad
                param.data -= self.lr * (param.grad + self.weight_decay * param.data)

    def zero_grad(self):
        for param in self.params:
            param.grad = None
