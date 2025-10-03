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
            # if param.grad is not None:
                param.data = -self.lr * param.grad
 
    def zero_grad(self):
        for param in self.params:
            param.grad = None
            # print("zero grad: ", param)
            # print("param.grad:", param.grad)
        return # a model that has None as gradient       return               # print(param.grad)
