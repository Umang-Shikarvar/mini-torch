import numpy as np
from minitorch.nn.parameter import Parameter
from minitorch.nn.modules.module import Module

class SGD:
    def __init__(self, params, lr=0.01) -> None:
        self.params = list(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            if not hasattr(param, 'data'):
                raise TypeError(f"Parameter {param} does not have 'data' attribute.")
            param.data = param.data - self.lr * param.grad  # SGD update

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad = np.zeros_like(param.data)  # fix: was param.graad = 0.0

    def __repr__(self):
        return f"SGD(lr={self.lr}) with parameters: {[param for param in self.params]}"


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8) -> None:
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(param.data) for param in self.params]
        self.v = [np.zeros_like(param.data) for param in self.params]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad = np.zeros_like(param.data)

    def __repr__(self):
        return f"Adam(lr={self.lr}, betas={self.betas}, eps={self.eps})"
