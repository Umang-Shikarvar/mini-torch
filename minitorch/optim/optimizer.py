from minitorch.nn.parameter import Parameter
from minitorch.nn.modules.module import Module

class SGD:
    
    def __init__(self, params, lr = 0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue

            if not hasattr(param, 'data'):
                raise TypeError(f"Parameter {param} does not have 'data' attribute.")
            
            param.data = param.data - self.lr * param.grad # update rule

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.graad = 0.0 

    def __repr__(self):
        return f"SGD(lr={self.lr}) with parameters: {[param for param in self.params]}"
    
