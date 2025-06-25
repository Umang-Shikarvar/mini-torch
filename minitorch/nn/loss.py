import numpy as np

class MSELoss:
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        if predictions.shape != targets.shape:
            raise ValueError("Shape mismatch between predictions and targets.")
        return ((predictions - targets)**2).mean()

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)


class BCELoss:
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        if predictions.shape != targets.shape:
            raise ValueError("Shape mismatch between predictions and targets.")
        predictions = np.clip(predictions, 1e-12, 1 - 1e-12) # to avoid log(0)
        return -(targets*np.log(predictions) + (1 - targets)*np.log(1 - predictions)).mean()

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, logits, class_indices):
        if logits.ndim != 2 or class_indices.ndim != 1 or logits.shape[0] != class_indices.shape[0]:
            raise ValueError("Invalid shapes for logits or class indices.")
        
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # softmax
        correct_class_probs = probabilities[np.arange(len(class_indices)), class_indices]
        return -np.log(correct_class_probs).mean()

    def __call__(self, logits, class_indices):
        return self.forward(logits, class_indices)
