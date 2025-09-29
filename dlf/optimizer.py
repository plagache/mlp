from .tensor import Tensor

def get_parameters(model):
    variables = vars(model)
    print("variables: ", variables)
    return variables


def zero_grad(params):
    for param in params:
        param.grad = None
        # print("zero grad: ", param)
        # print("param.grad:", param.grad)
    return # a model that has None as gradient       return

class SGD():
    def __init__(self, params:list[Tensor], learning_rate: float):
        self.params = params
        self.lr = learning_rate

    def step(self):
        zero_grad(self.params)
        for param in self.params:
            if param.grad is not None:
                param.data = Tensor(-self.lr) * param.grad
                print(param.data)
                # print(param)
                # print(param.grad)
