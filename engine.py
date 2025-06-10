import numpy as np

class Tensor:
    def __init__(self, data, children=()):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self.children = set(children)

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    def __add__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        try:
            out = Tensor(self.data + other.data, children=(self, other))
        except ValueError as e:
            raise ValueError(f"Tensor shapes {self.data.shape} and {other.data.shape} are not compatible for addition.") from e

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out       
    
a= Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
c = a + b
print(c)  