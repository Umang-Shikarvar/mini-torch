import numpy as np

def _handle_broadcasting(grad, target_shape):
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad

class Tensor:
    def __init__(self, data, children=(), _op='', requires_grad= True):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data, dtype=float)
        self._backward = lambda: None
        self.children = set(children)
        self._op = _op
        self.requires_grad = requires_grad

    # def __repr__(self):
    #     return f"tensor(data={self.data}, grad={self.grad})"

    def item(self):
        if self.data.size != 1:
            raise ValueError("Cannot convert a tensor with more than one element to a scalar.")
        return self.data.item()

    def __repr__(self):
        data_str = np.round(self.data, 4)
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"tensor({data_str}{grad_str})"

    
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
                grad_self = _handle_broadcasting(grad_self, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = out.grad
                grad_other = _handle_broadcasting(grad_other, other.data.shape)
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
                grad_self = _handle_broadcasting(grad_self, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = out.grad * self.data
                grad_other = _handle_broadcasting(grad_other, other.data.shape)
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
                grad_self = _handle_broadcasting(grad_self, self.data.shape)
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
        out = Tensor(out_data, children=(self, other), _op='@', requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
                grad_self = _handle_broadcasting(grad_self, self.data.shape)
                self.grad += grad_self

            if other.requires_grad:
                grad_other = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
                grad_other = _handle_broadcasting(grad_other, other.data.shape)
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



def pow(tensor, power):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    return tensor ** power
Tensor.pow = staticmethod(pow)


def exp(tensor):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    out = Tensor(np.exp(tensor.data), children=(tensor,), _op='exp', requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:
            grad_self = out.grad * np.exp(tensor.data)
            grad_self = _handle_broadcasting(grad_self, tensor.data.shape)
            tensor.grad += grad_self
    out._backward = _backward

    return out
Tensor.exp = exp


def log(tensor):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    if np.any(tensor.data < 0):
        print("Warning: log of negative number will result in NaN.")

    out = Tensor(np.log(tensor.data), children=(tensor,), _op='log', requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:
            grad_self = out.grad / tensor.data
            grad_self = _handle_broadcasting(grad_self, tensor.data.shape)
            tensor.grad += grad_self
    out._backward = _backward

    return out
Tensor.log = log

