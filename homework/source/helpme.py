import numpy as np


def eval_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-10, np.abs(a) * np.abs(b)))
    
    
def numerical_grad_array(f, x, df, h=1e-5):
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    grad = np.zeros_like(x)
    while not it.finished:
        idx = it.multi_index
        
        it[0] += h
        f_pos = f(x)
        
        it[0] -= 2*h
        f_neg = f(x)
        
        it[0] += h
        
        grad[idx] += np.sum((f_pos - f_neg) * df) / (2 * h)
        it.iternext()
    return grad 