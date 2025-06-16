import minitorch

class Parameter(minitorch.Tensor):
    def __hash__(self):
        return id(self)
    
    def __init__(self, tensor):
        if not isinstance(tensor, minitorch.Tensor):
             tensor = minitorch.Tensor(tensor) # Ensure tensor is a Tensor instance
             
        super().__init__(tensor.data)
        self.requires_grad = True

    def zero_grad(self):
        if hasattr(self, 'grad'):
            self.grad = minitorch.Tensor(0.0, requires_grad=False)
        else:
            raise AttributeError("Parameter does not have a 'grad' attribute to zero out.")

    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"
