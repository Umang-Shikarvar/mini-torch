import random
from minitorch.engine import Tensor
from .module import Module
from ..parameter import Parameter

class Linear(Module):
    def __init__(self,in_features, out_features, bias=True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k= 1 / (in_features ** 0.5)
        
        weight_data = [[random.uniform(-k, k) for _ in range(out_features)] for _ in range(in_features)]
        self.weight = Parameter(Tensor(weight_data)) 

        if bias:
            bias_data = [random.uniform(-k, k) for _ in range(out_features)]
            self.bias = Parameter(Tensor(bias_data))
        else:
            self.bias = None

    
    def forward(self, x):
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")
        if x.data.shape[1] != self.in_features:
            raise ValueError(f"Input shape {x.data.shape} does not match expected shape ({x.data.shape[0]}, {self.in_features}).")
        
        out = x @ self.weight
        if self.bias is not None:
            out += self.bias
        return out
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"