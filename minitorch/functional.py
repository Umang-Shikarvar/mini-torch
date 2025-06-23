# functional.py
import numpy as np
from minitorch.engine import Tensor, _handle_broadcasting

def sum(tensor, axis=None, keepdims=False):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    out = Tensor(np.sum(tensor.data, axis=axis, keepdims=keepdims), children=(tensor,), _op=f'sum(axis={axis},keepdims={keepdims})', requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:
            grad_self = out.grad
            # Broadcast grad_self to the shape of tensor.data
            if axis is not None:
                shape = list(tensor.data.shape)
                if not isinstance(axis, tuple):
                    axes = (axis,)
                else:
                    axes = axis
                for ax in sorted(axes):
                    shape[ax] = 1
                grad_self = np.reshape(grad_self, shape)
            grad_self = np.broadcast_to(grad_self, tensor.data.shape)
            tensor.grad += grad_self
    out._backward = _backward
    return out

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

def relu(tensor):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    relu_val = np.maximum(0, tensor.data)
    out = Tensor(relu_val, children=(tensor,), _op='relu', requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:
            mask = (tensor.data > 0).astype(tensor.data.dtype)
            grad_self = out.grad * mask
            grad_self = _handle_broadcasting(grad_self, tensor.data.shape)
            tensor.grad += grad_self
    out._backward = _backward

    return out

def sigmoid(tensor):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    sig_val = 1 / (1 + np.exp(-tensor.data))
    out = Tensor(sig_val, children=(tensor,), _op='sigmoid', requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:
            grad_self = out.grad * out.data * (1 - out.data)
            grad_self = _handle_broadcasting(grad_self, tensor.data.shape)
            tensor.grad += grad_self
    out._backward = _backward

    return out

def tanh(tensor):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    tanh_val = np.tanh(tensor.data)
    out = Tensor(tanh_val, children=(tensor,), _op='tanh', requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:
            grad_self = out.grad * (1 - out.data ** 2)
            grad_self = _handle_broadcasting(grad_self, tensor.data.shape)
            tensor.grad += grad_self
    out._backward = _backward

    return out

def leaky_relu(tensor, negative_slope=0.01):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    leaky_relu_val = np.where(tensor.data > 0, tensor.data, negative_slope * tensor.data)
    out = Tensor(leaky_relu_val, children=(tensor,), _op='leaky_relu', requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:
            mask = (tensor.data > 0).astype(tensor.data.dtype) + negative_slope * (tensor.data <= 0).astype(tensor.data.dtype)
            grad_self = out.grad * mask
            grad_self = _handle_broadcasting(grad_self, tensor.data.shape)
            tensor.grad += grad_self
    out._backward = _backward

    return out

def softmax(tensor, dim=-1):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    shifted = tensor.data - np.max(tensor.data, axis=dim, keepdims=True)
    exps = np.exp(shifted)
    softmax_val = exps / np.sum(exps, axis=dim, keepdims=True)
    out = Tensor(softmax_val, children=(tensor,), _op='softmax', requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:

            ## previous code using jacobian
            # grad_self = np.empty_like(out.data)
            # for i in range(out.data.shape[0]):
            #     s = out.data[i].reshape(-1, 1)
            #     jacobian = np.diagflat(s) - np.dot(s, s.T)
            #     grad_self[i] = np.dot(jacobian, out.grad[i])


            # grad shape: same as out.data
            grad = out.grad
            s = out.data
            # sum over the softmax axis
            dot = np.sum(grad * s, axis=dim, keepdims=True)
            grad_self = s * (grad - dot)
            grad_self = _handle_broadcasting(grad_self, tensor.data.shape)
            tensor.grad += grad_self
            
    out._backward = _backward

    return out
