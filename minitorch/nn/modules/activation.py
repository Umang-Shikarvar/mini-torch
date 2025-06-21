from minitorch.nn.modules.module import Module
import minitorch.functional as F
from minitorch.engine import Tensor

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input)

    def __repr__(self):
        return "ReLU()"


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, negative_slope=self.negative_slope)

    def __repr__(self):
        return f"LeakyReLU(negative_slope={self.negative_slope})"

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input)

    def __repr__(self):
        return "Sigmoid()"

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.tanh(input)

    def __repr__(self):
        return "Tanh()"

class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.softmax(input, dim=self.dim)

    def __repr__(self):
        return f"Softmax(dim={self.dim})"