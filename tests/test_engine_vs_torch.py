import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import numpy as np
import pytest
from minitorch import Tensor, exp, log

def assert_tensor_allclose(t1, t2, atol=1e-6):
    assert np.allclose(np.array(t1.data), np.array(t2.detach().cpu().numpy()), atol=atol)
    if hasattr(t1, 'grad') and t1.grad is not None and t2.grad is not None:
        assert np.allclose(np.array(t1.grad), np.array(t2.grad.detach().cpu().numpy()), atol=atol)

def test_add():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a + b
    t_a = torch.tensor(2.0, requires_grad=True)
    t_b = torch.tensor(3.0, requires_grad=True)
    t_c = t_a + t_b
    assert_tensor_allclose(c, t_c)
    c.backward()
    t_c.backward()
    assert_tensor_allclose(a, t_a)
    assert_tensor_allclose(b, t_b)

def test_mul():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a * b
    t_a = torch.tensor(2.0, requires_grad=True)
    t_b = torch.tensor(3.0, requires_grad=True)
    t_c = t_a * t_b
    assert_tensor_allclose(c, t_c)
    c.backward()
    t_c.backward()
    assert_tensor_allclose(a, t_a)
    assert_tensor_allclose(b, t_b)

def test_pow():
    a = Tensor(2.0)
    c = a ** 3
    t_a = torch.tensor(2.0, requires_grad=True)
    t_c = t_a ** 3
    assert_tensor_allclose(c, t_c)
    c.backward()
    t_c.backward()
    assert_tensor_allclose(a, t_a)

def test_exp():
    a = Tensor(1.5)
    c = a.exp()
    t_a = torch.tensor(1.5, requires_grad=True)
    t_c = t_a.exp()
    assert_tensor_allclose(c, t_c)
    c.backward()
    t_c.backward()
    assert_tensor_allclose(a, t_a)

def test_log():
    a = Tensor(2.0)
    c = a.log()
    t_a = torch.tensor(2.0, requires_grad=True)
    t_c = t_a.log()
    assert_tensor_allclose(c, t_c)
    c.backward()
    t_c.backward()
    assert_tensor_allclose(a, t_a)

def test_neg():
    a = Tensor(2.0)
    c = -a
    t_a = torch.tensor(2.0, requires_grad=True)
    t_c = -t_a
    assert_tensor_allclose(c, t_c)
    c.backward()
    t_c.backward()
    assert_tensor_allclose(a, t_a)

def test_sub():
    a = Tensor(5.0)
    b = Tensor(3.0)
    c = a - b
    t_a = torch.tensor(5.0, requires_grad=True)
    t_b = torch.tensor(3.0, requires_grad=True)
    t_c = t_a - t_b
    assert_tensor_allclose(c, t_c)
    c.backward()
    t_c.backward()
    assert_tensor_allclose(a, t_a)
    assert_tensor_allclose(b, t_b)

def test_truediv():
    a = Tensor(6.0)
    b = Tensor(2.0)
    c = a / b
    t_a = torch.tensor(6.0, requires_grad=True)
    t_b = torch.tensor(2.0, requires_grad=True)
    t_c = t_a / t_b
    assert_tensor_allclose(c, t_c)
    c.backward()
    t_c.backward()
    assert_tensor_allclose(a, t_a)
    assert_tensor_allclose(b, t_b)

def test_matmul():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[2.0, 0.0], [1.0, 2.0]])
    c = a @ b
    t_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    t_b = torch.tensor([[2.0, 0.0], [1.0, 2.0]], requires_grad=True)
    t_c = t_a @ t_b
    assert_tensor_allclose(c, t_c)
    c.backward(np.ones_like(c.data))
    t_c.backward(torch.ones_like(t_c))
    assert_tensor_allclose(a, t_a)
    assert_tensor_allclose(b, t_b)

def test_radd():
    a = Tensor(2.0)
    c = 3.0 + a
    t_a = torch.tensor(2.0, requires_grad=True)
    t_c = 3.0 + t_a
    assert_tensor_allclose(c, t_c)
    c.backward()
    t_c.backward()
    assert_tensor_allclose(a, t_a)

def test_rsub():
    a = Tensor(2.0)
    c = 3.0 - a
    t_a = torch.tensor(2.0, requires_grad=True)
    t_c = 3.0 - t_a
    assert_tensor_allclose(c, t_c)
    c.backward()
    t_c.backward()
    assert_tensor_allclose(a, t_a)

def test_rmul():
    a = Tensor(2.0)
    c = 3.0 * a
    t_a = torch.tensor(2.0, requires_grad=True)
    t_c = 3.0 * t_a
    assert_tensor_allclose(c, t_c)
    c.backward()
    t_c.backward()
    assert_tensor_allclose(a, t_a)

def test_rtruediv():
    a = Tensor(2.0)
    c = 6.0 / a
    t_a = torch.tensor(2.0, requires_grad=True)
    t_c = 6.0 / t_a
    assert_tensor_allclose(c, t_c)
    c.backward()
    t_c.backward()
    assert_tensor_allclose(a, t_a)

def test_exp_function():
    a = Tensor(1.5)
    c = exp(a)
    t_a = torch.tensor(1.5, requires_grad=True)
    t_c = t_a.exp()
    assert_tensor_allclose(c, t_c)
    c.backward()
    t_c.backward()
    assert_tensor_allclose(a, t_a)

def test_log_function():
    a = Tensor(2.0)
    c = log(a)
    t_a = torch.tensor(2.0, requires_grad=True)
    t_c = t_a.log()
    assert_tensor_allclose(c, t_c)
    c.backward()
    t_c.backward()
    assert_tensor_allclose(a, t_a)
