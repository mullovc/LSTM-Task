import numpy as np

def sigmoid(x):
    return np.divide(1.0, 1.0 + np.exp(-x))

def cosine(x, y):
    if x.ndim == 1:
        return np.divide(np.dot(x, y.transpose()),
                         np.linalg.norm(x) * np.linalg.norm(y))
    elif x.ndim == 2:
        return np.mean([cosine(x_i, y_i) for x_i, y_i in zip(x, y)])
