from minitorch.engine import Tensor

class Parameter(Tensor):
    def __init__(self, tensor):
        super().__init__(tensor.data)
        self.requires_grad = True

    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"
