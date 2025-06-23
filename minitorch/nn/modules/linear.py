import random
import minitorch
from minitorch.nn.parameter import Parameter

from .module import Module

import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k= 1 / (in_features ** 0.5)
        
        # weight_data = [[random.uniform(-k, k) for _ in range(in_features)] for _ in range(out_features)]
        weight_data = np.random.uniform(-k, k, size=(out_features, in_features)).tolist()
        self.weight = Parameter(minitorch.Tensor(weight_data), name='weight') 

        if bias:
            # bias_data = [random.uniform(-k, k) for _ in range(out_features)]
            bias_data = np.random.uniform(-k, k, size=(out_features)).tolist()
            self.bias = Parameter(minitorch.Tensor(bias_data), name='bias')
        else:
            self.bias = None

    def forward(self, x):
        if not isinstance(x, minitorch.Tensor):
            raise TypeError("Input must be a Tensor.")
        if x.data.shape[1] != self.in_features:
            raise ValueError(f"Input shape {x.data.shape} does not match expected shape ({x.data.shape[0]}, {self.in_features}).")
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

