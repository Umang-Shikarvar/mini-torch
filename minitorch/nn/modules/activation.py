from minitorch.nn.modules.module import Module
from minitorch.functional import relu as functional_relu
from minitorch.functional import sigmoid as functional_sigmoid
from minitorch.functional import tanh as functional_tanh
from minitorch.engine import Tensor

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return functional_relu(input)

    def __repr__(self):
        return "ReLU()"

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return functional_sigmoid(input)

    def __repr__(self):
        return "Sigmoid()"

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return functional_tanh(input)

    def __repr__(self):
        return "Tanh()"