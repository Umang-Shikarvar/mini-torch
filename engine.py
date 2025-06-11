import numpy as np

class Tensor:
    def __init__(self, data, children=(), _op='', requires_grad=True):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data, dtype=float)
        self._backward = lambda: None
        self.children = set(children)
        self._op = _op
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    @property
    def shape(self):
        return self.data.shape

    

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        requires_grad = self.requires_grad or other.requires_grad
        try:
            out = Tensor(self.data + other.data, children=(self, other), _op='+',requires_grad=requires_grad)
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
            out = Tensor(self.data * other.data, children=(self, other), _op='*',requires_grad=requires_grad)
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


    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("Power must be an integer or float.")
        
        out = Tensor(self.data ** power, children=(self,), _op='**', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * power * (self.data ** (power - 1))
                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0)
                for axis, size in enumerate(self.data.shape):
                    if size == 1:
                        grad_self = grad_self.sum(axis=axis, keepdims=True)
                self.grad += grad_self
        out._backward = _backward

        return out


    def exp(self):
        out = Tensor(np.exp(self.data), children=(self,), _op='exp', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * np.exp(self.data)
                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0)
                for axis, size in enumerate(self.data.shape):
                    if size == 1:
                        grad_self = grad_self.sum(axis=axis, keepdims=True)
                self.grad += grad_self
        out._backward = _backward

        return out
    

    def log(self):
        if np.any(self.data <= 0):
            raise ValueError("Logarithm undefined for non-positive values.")
        
        out = Tensor(np.log(self.data), children=(self,), _op='log', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad / self.data
                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0)
                for axis, size in enumerate(self.data.shape):
                    if size == 1:
                        grad_self = grad_self.sum(axis=axis, keepdims=True)
                self.grad += grad_self
        out._backward = _backward

        return out
    
    
    def __neg__(self):      # -self
        return self * -1
    
    def __sub__(self, other):       # self - other
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)
    
    def __radd__(self, other):      # other + self
        return self+other
    
    def __rsub__(self, other):      # other - self
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other + (-self)
    
    def __rmul__(self, other):      # other * self
        return self * other

    def __truediv__(self, other):       # self / other
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):      # other / self
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other * (self ** -1)
    

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        requires_grad = self.requires_grad or other.requires_grad
        out_data = np.matmul(self.data, other.data)
        out = Tensor(out_data, children=(self, other), requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
                # Handle broadcasting during backprop
                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0)
                for axis, size in enumerate(self.data.shape):
                    if size == 1:
                        grad_self = grad_self.sum(axis=axis, keepdims=True)
                self.grad += grad_self

            if other.requires_grad:
                grad_other = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
                while grad_other.ndim > other.data.ndim:
                    grad_other = grad_other.sum(axis=0)
                for axis, size in enumerate(other.data.shape):
                    if size == 1:
                        grad_other = grad_other.sum(axis=axis, keepdims=True)
                other.grad += grad_other

        out._backward = _backward
        return out

    
    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")
        if grad is None:
            grad = np.ones_like(self.data, dtype=float)
        else:
            grad = np.array(grad, dtype=float)
        self.grad = grad

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