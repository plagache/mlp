from .tensor import Tensor

def get_parameters(model):
    variables = vars(model)
    return variables

def zero_grad(model):
    return # a model that has None as gradient

class SGD():
    def __init__(self, learning_rate: float):
        return
