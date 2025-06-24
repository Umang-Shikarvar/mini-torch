import numpy as np

class MSELoss:
    def __init__(self) -> None:
        pass

    def forward(self, input, target):
        # input and target should be Tensor
        if input.shape != target.shape:
            raise ValueError(f"Input shape {input.shape} does not match target shape {target.shape}.")
        return ((input - target) ** 2).mean()

    def __call__(self, input, target):
        return self.forward(input, target)
    

class BCEloss:
    def __init__(self) -> None:
        pass

    def forward(self, y_pred, y_target):
        # y_pred and y_target should be Tensor
        if y_pred.shape != y_target.shape:
            raise ValueError(f"y_pred shape {y_pred.shape} does not match y_target shape {y_target.shape}.")
        epsilon = 1e-12
        # Clamp predictions to avoid log(0) using Tensor ops
        y_pred_clamped = y_pred * (1 - 2 * epsilon) + epsilon
        loss = -(y_target * y_pred_clamped.log() + (1 - y_target) * (1 - y_pred_clamped).log()).mean()
        return loss

    def __call__(self, y_pred, y_target):
        return self.forward(y_pred, y_target)