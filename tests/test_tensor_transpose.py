import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pytest
import minitorch

# Helper

def assert_tensor_allclose(t1, t2, atol=1e-6):
    assert np.allclose(np.array(t1.data), np.array(t2.detach().cpu().numpy()), atol=atol)
    if hasattr(t1, 'grad') and t1.grad is not None and t2.grad is not None:
        assert np.allclose(np.array(t1.grad), np.array(t2.grad.detach().cpu().numpy()), atol=atol)

def test_T_2d():
    x_np = np.random.randn(2, 3)
    x = minitorch.Tensor(x_np)
    x_torch = torch.tensor(x_np, requires_grad=True)
    y = x.T
    y_torch = x_torch.T
    assert_tensor_allclose(y, y_torch)
    y.backward(np.ones_like(y.data))
    y_torch.backward(torch.ones_like(y_torch))
    assert_tensor_allclose(x, x_torch)

def test_transpose_2d():
    x_np = np.random.randn(2, 3)
    x = minitorch.Tensor(x_np)
    x_torch = torch.tensor(x_np, requires_grad=True)
    y = x.transpose(0, 1)
    y_torch = x_torch.transpose(0, 1)
    assert_tensor_allclose(y, y_torch)
    y.backward(np.ones_like(y.data))
    y_torch.backward(torch.ones_like(y_torch))
    assert_tensor_allclose(x, x_torch)

def test_transpose_3d():
    x_np = np.random.randn(2, 3, 4)
    x = minitorch.Tensor(x_np)
    x_torch = torch.tensor(x_np, requires_grad=True)
    y = x.transpose(0, 2)
    y_torch = x_torch.transpose(0, 2)
    assert_tensor_allclose(y, y_torch)
    y.backward(np.ones_like(y.data))
    y_torch.backward(torch.ones_like(y_torch))
    assert_tensor_allclose(x, x_torch)

def test_transpose_multiple():
    x_np = np.random.randn(2, 3, 4)
    x = minitorch.Tensor(x_np)
    x_torch = torch.tensor(x_np, requires_grad=True)
    y = x.transpose(0, 2).transpose(1, 2)
    y_torch = x_torch.transpose(0, 2).transpose(1, 2)
    assert_tensor_allclose(y, y_torch)
    y.backward(np.ones_like(y.data))
    y_torch.backward(torch.ones_like(y_torch))
    assert_tensor_allclose(x, x_torch)

def test_minitorch_transpose_func():
    x_np = np.random.randn(2, 3, 4)
    x = minitorch.Tensor(x_np)
    x_torch = torch.tensor(x_np, requires_grad=True)
    y = minitorch.transpose(x, 0, 2)
    y_torch = x_torch.transpose(0, 2)
    assert_tensor_allclose(y, y_torch)
    y.backward(np.ones_like(y.data))
    y_torch.backward(torch.ones_like(y_torch))
    assert_tensor_allclose(x, x_torch)
