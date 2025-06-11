import numpy as np

class Tensor:
    def __init__(self, data, children=(), _op=''):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self.children = set(children)
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        try:
            out = Tensor(self.data + other.data, children=(self, other), _op='+')
        except ValueError as e:
            raise ValueError(f"Tensor shapes {self.data.shape} and {other.data.shape} are not compatible for addition.") from e

        def _backward():
            self_grad = out.grad
            other_grad = out.grad
            while self_grad.ndim > self.data.ndim:
                self_grad = self_grad.sum(axis=0)
            for  axis,size in enumerate(self.data.shape):
                if size == 1:
                    self_grad = self_grad.sum(axis=axis, keepdims=True)
            while other_grad.ndim > other.data.ndim:
                other_grad = other_grad.sum(axis=0)
            for  axis,size in enumerate(other.data.shape):
                if size == 1:
                    other_grad = other_grad.sum(axis=axis, keepdims=True)
            self.grad += self_grad
            other.grad += other_grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self+other
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        try:
            out = Tensor(self.data * other.data, children=(self, other), _op='*')
        except ValueError as e:
            raise ValueError(f"Tensor shapes {self.data.shape} and {other.data.shape} are not compatible for addition.") from e

        def _backward():
            self_grad = out.grad * other.data
            other_grad = out.grad * self.data
            while self_grad.ndim > self.data.ndim:
                self_grad = self_grad.sum(axis=0)
            for  axis,size in enumerate(self.data.shape):
                if size == 1:
                    self_grad = self_grad.sum(axis=axis, keepdims=True)
            while other_grad.ndim > other.data.ndim:
                other_grad = other_grad.sum(axis=0)
            for  axis,size in enumerate(other.data.shape):
                if size == 1:
                    other_grad = other_grad.sum(axis=axis, keepdims=True)
            self.grad += self_grad
            other.grad += other_grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other
    
    def backward(self):
        self.grad= np.ones_like(self.data)

        topo= []
        visited = set()
        def topo_sort(tensor):
            if(tensor not in visited):
                visited.add(tensor)
                for child in tensor.children:
                    topo_sort(child)
                topo.append(tensor)

        topo_sort(self)
        for tensor in reversed(topo):
            tensor._backward()

    def zero_grad(self):
        topo= []
        visited = set()
        def topo_sort(tensor):
            if(tensor not in visited):
                visited.add(tensor)
                for child in tensor.children:
                    topo_sort(child)
                topo.append(tensor)

        topo_sort(self)
        for tensor in topo:
            tensor.grad = np.zeros_like(tensor.data)