import numpy as np

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
    

class BCEloss:
    def __init__(self) -> None:
        pass

    def forward(self, y_pred, y_target):
        if y_pred.shape != y_target.shape:
            raise ValueError(f"y_pred shape {y_pred.shape} does not match y_target shape {y_target.shape}.")
        
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Prevent log(0)
        bce = -np.mean(y_target * np.log(y_pred) + (1 - y_target) * np.log(1 - y_pred))
        return bce
        
    def backward(self, y_pred, y_target):
        if y_pred.shape != y_target.shape:
            raise ValueError(f"y_pred shape {y_pred.shape} does not match y_target shape {y_target.shape}.")
        
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        grad = -(y_target / y_pred - (1 - y_target) / (1 - y_pred)) / y_pred.size
        return grad

    def __call__(self, y_pred, y_target) -> float:
        return self.forward(y_pred, y_target)