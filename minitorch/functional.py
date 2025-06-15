# functional.py
import numpy as np
from minitorch.engine import Tensor, _handle_broadcasting

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

def pow(tensor, power):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    return tensor ** power

def transpose(tensor, dim0, dim1):
    # Swap two axes and return a new Tensor with autograd support
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    axes = list(range(tensor.data.ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    out = Tensor(tensor.data.transpose(axes), children=(tensor,), _op=f'transpose({dim0},{dim1})', requires_grad=tensor.requires_grad)
    def _backward():
        if tensor.requires_grad:
            # Reverse the axes swap for the gradient
            reverse_axes = list(range(tensor.data.ndim))
            reverse_axes[dim0], reverse_axes[dim1] = reverse_axes[dim1], reverse_axes[dim0]
            grad_self = out.grad.transpose(reverse_axes)
            grad_self = _handle_broadcasting(grad_self, tensor.data.shape)
            tensor.grad += grad_self
    out._backward = _backward
    return out
