import random
from engine import Tensor

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return []
    
    def __call__(self,x):
        return self.forward(x)
    
    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in subclasses.")

class Linear(Module):
    def __init__(self,in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        k= 1 / (in_features ** 0.5)
        self.weight = Tensor([[random.uniform(-k, k) for _ in range(out_features)] for _ in range(in_features)])
        self.bias = Tensor([random.uniform(-k, k) for _ in range(out_features)]) if bias else None

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
    
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