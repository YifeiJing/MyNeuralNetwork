import numpy as np

def mean_squared_loss(x, y):
    if x.ndim == 2:
        return np.sum(np.sum((x - y)**2, axis=1)) / x.shape[0]
    
    return np.sum((x - y)**2)

def cross_entropy_loss(x, y):
    if x.ndim == 2:
        return np.sum(-np.sum(y * np.log(x + 1e-7), axis=1)) / x.shape[0]
    
    return -np.sum(y * np.log(x + 1e-7))