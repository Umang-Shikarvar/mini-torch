import numpy as np
import minitorch

def _handle_broadcasting(grad, target_shape):
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad

class Tensor:

    def __hash__(self):
        return id(self)
    
    def __init__(self, data, children=(), _op='', requires_grad=True, dtype=None):
        # Use minitorch.float32 and minitorch.bool as canonical dtypes
        valid_dtypes = {minitorch.float32, minitorch.bool, float, bool, 'float32', 'bool'}
        def resolve_dtype(dt):
            if dt is None:
                return None
            if dt is minitorch.float32 or dt is float or dt == 'float32':
                return minitorch.float32
            if dt is minitorch.bool or dt is bool or dt == 'bool':
                return minitorch.bool
            raise TypeError(f"Only float32 and bool dtypes are supported, got {dt}.")

        resolved_dtype = resolve_dtype(dtype)

        # If data is already a numpy array
        if isinstance(data, np.ndarray):
            if resolved_dtype is not None:
                self.data = data.astype(resolved_dtype)
            else:
                if data.dtype == minitorch.bool:
                    self.data = data.astype(minitorch.bool)
                else:
                    self.data = data.astype(minitorch.float32)
        else:
            if resolved_dtype is not None:
                self.data = np.array(data, dtype=resolved_dtype)
            else:
                temp = np.array(data)
                if temp.dtype == minitorch.bool:
                    self.data = temp.astype(minitorch.bool)
                else:
                    self.data = temp.astype(minitorch.float32)
        self._dtype = self.data.dtype
        if self._dtype == minitorch.float32:
            self.grad = np.zeros_like(self.data, dtype=self._dtype)
        else:
            self.grad = None
        self._backward = lambda: None
        self.children = set(children)
        self._op = _op
        self.requires_grad = requires_grad
    
    def item(self):
        if self.data.size != 1:
            raise ValueError("Cannot convert a tensor with more than one element to a scalar.")
        return self.data.item()

    def __repr__(self):
        if np.issubdtype(self._dtype, np.floating):
            data_str = np.array2string(self.data, precision=4, separator=', ', suppress_small=False)
        else:
            data_str = np.array2string(self.data, separator=', ')
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"tensor({data_str}{grad_str})"

    def __getitem__(self, idx):
        # Support slicing/indexing, return a new Tensor with autograd tracking
        out_data = self.data[idx]
        out = Tensor(out_data, children=(self,), _op=f'getitem[{idx}]', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                grad_self = np.zeros_like(self.data, dtype=self.data.dtype)
                grad_self[idx] = out.grad
                grad_self = _handle_broadcasting(grad_self, self.data.shape)
                self.grad += grad_self
        out._backward = _backward
        return out

    def __setitem__(self, idx, value):
        """
        Set item(s) in the underlying data array. This operation is in-place and breaks the computation graph.
        Autograd will not track this assignment. Use with caution.
        """
        if isinstance(value, Tensor):
            value = value.data
        self.data[idx] = value
        # If gradients exist, reset them for the modified indices
        if self.grad is not None:
            self.grad[idx] = 0
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        if self._dtype == minitorch.float32:
            return 'minitorch.float32'
        elif self._dtype == minitorch.bool:
            return 'minitorch.bool'
        else:
            raise TypeError(f"Tensor dtype {self._dtype} is not supported.")
    
    @property
    def T(self):
        out = Tensor(self.data.T, children=(self,), _op='T', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                grad_self = out.grad.T
                grad_self = _handle_broadcasting(grad_self, self.data.shape)
                self.grad += grad_self
        out._backward = _backward
        return out
    
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

    def __lt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data < other.data, children=(self, other), _op='lt', requires_grad=False, dtype=minitorch.bool)

    def __le__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data <= other.data, children=(self, other), _op='le', requires_grad=False, dtype=minitorch.bool)

    def __gt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data > other.data, children=(self, other), _op='gt', requires_grad=False, dtype=minitorch.bool)

    def __ge__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data >= other.data, children=(self, other), _op='ge', requires_grad=False, dtype=minitorch.bool)

    def __eq__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data == other.data, children=(self, other), _op='eq', requires_grad=False, dtype=minitorch.bool)

    def __ne__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data != other.data, children=(self, other), _op='ne', requires_grad=False, dtype=minitorch.bool)
    
    
    # Functions from functional.py

    def sum(self, axis=None, keepdims=False):
        from .functional import sum
        return sum(self, axis=axis, keepdims=keepdims)
    
    def exp(self):
        from .functional import exp
        return exp(self)

    def log(self):
        from .functional import log
        return log(self)

    def pow(self, power):
        from .functional import pow
        return pow(self, power)
    
    def transpose(self, dim0, dim1):
        from .functional import transpose
        return transpose(self, dim0, dim1)
    
    def relu(self):
        from .functional import relu
        return relu(self)

    def leaky_relu(self, negative_slope=0.01):
        from .functional import leaky_relu
        return leaky_relu(self, negative_slope=negative_slope)

    def sigmoid(self):
        from .functional import sigmoid
        return sigmoid(self)

    def tanh(self):
        from .functional import tanh
        return tanh(self)

    def softmax(self, dim=-1):
        from .functional import softmax
        return softmax(self, dim=dim)

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
        topo = []
        visited = set()
        def topo_sort(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor.children:
                    topo_sort(child)
                topo.append(tensor)

        topo_sort(self)
        for tensor in topo:
            tensor.grad = np.zeros_like(tensor.data)

    def mean(self):
        if self.data.size == 0:
            raise ValueError("cannot compute mean of an empty tensor")
        
        mean_value = np.mean(self.data)
        out = Tensor(mean_value, children=(self,), _op='mean', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                grad_self = out.grad * np.ones_like(self.data) / self.data.size
                grad_self = _handle_broadcasting(grad_self, self.data.shape)
                self.grad += grad_self
        out._backward = _backward
        return out
