import numpy as np

class Tensor:
    def __init__(self, data, children=(), requires_grad=True):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self.children = set(children)
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        requires_grad = self.requires_grad or other.requires_grad
        try:
            out = Tensor(self.data + other.data, children=(self, other), requires_grad=requires_grad)
        except ValueError as e:
            raise ValueError(f"Tensor shapes {self.data.shape} and {other.data.shape} are not compatible for addition.") from e

        def _backward():
            if self.requires_grad:
                grad_self = out.grad
                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0)
                for axis, size in enumerate(self.data.shape):
                    if size == 1:
                        grad_self = grad_self.sum(axis=axis, keepdims=True)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = out.grad
                while grad_other.ndim > other.data.ndim:
                    grad_other = grad_other.sum(axis=0)
                for axis, size in enumerate(other.data.shape):
                    if size == 1:
                        grad_other = grad_other.sum(axis=axis, keepdims=True)
                other.grad += grad_other
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        requires_grad = self.requires_grad or other.requires_grad
        try:
            out = Tensor(self.data * other.data, children=(self, other), requires_grad=requires_grad)
        except ValueError as e:
            raise ValueError(f"Tensor shapes {self.data.shape} and {other.data.shape} are not compatible for multiplication.") from e

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * other.data
                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0)
                for axis, size in enumerate(self.data.shape):
                    if size == 1:
                        grad_self = grad_self.sum(axis=axis, keepdims=True)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = out.grad * self.data
                while grad_other.ndim > other.data.ndim:
                    grad_other = grad_other.sum(axis=0)
                for axis, size in enumerate(other.data.shape):
                    if size == 1:
                        grad_other = grad_other.sum(axis=axis, keepdims=True)
                other.grad += grad_other
        out._backward = _backward

        return out





    
    def backward(self):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")
        self.grad = np.ones_like(self.data)

        topo = []
        visited = set()
        def topo_sort(tensor):
            if tensor not in visited:
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