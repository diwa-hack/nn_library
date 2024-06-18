import math

from engine import Value

def mse_loss(predictions, targets):
    loss = sum((p - t)**2 for p, t in zip(predictions, targets)) / len(targets)
    return loss

def binary_cross_entropy(predictions, targets):
    loss = -sum(t * Value(math.log(p.data)) + (1 - t) * Value(math.log(1 - p.data)) for p, t in zip(predictions, targets)) / len(targets)
    return loss
