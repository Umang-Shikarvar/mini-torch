
class MSELoss:
    def __init__(self) -> None:
        pass

    def forward(self, input, target):
        if input.shape != target.shape:
            raise ValueError(f"Input shape {input.shape} does not match target shape {target.shape}.")
        
        return ((input - target) ** 2).mean()
    
    def backward(self, input, target):
        if input.shape != target.shape:
            raise ValueError(f"Input shape {input.shape} does not match target shape {target.shape}.")
        
        grad = 2 * (input - target) / input.numel()
        return grad

    def __call__(self, input, target) -> float:
        return self.forward(input, target)
    