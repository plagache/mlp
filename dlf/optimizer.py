from .tensor import Tensor

# def get_parameters(model):
#     variables = vars(model)
#     print("variables: ", variables)
#     return variables



class SGD():
    def __init__(self, params:list[Tensor], learning_rate: float):
        self.params = params
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
