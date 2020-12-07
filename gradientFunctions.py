import numpy as np

def numerical_grad(f, x):
    """
    Compute the numerical gradient as point x
    """
    grad = np.zeros_like(x)
    h = 1e-5
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = float(tmp_val) - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        it.iternext()   
        
    return grad

if __name__ == '__main__':
    def f(x):
        return 2 * x
    x1 = np.array([[1.0]])
    print(numerical_grad(f, x1)) # 2.0