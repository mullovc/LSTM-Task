import numpy as np

class Module(object):
    def __init__(self):
        self.parameters = dict()
        self.gradients = dict()
        self.sub_modules = []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            if p not in self.gradients:
                self.gradients[p] = np.zeros_like(self.parameters[p])
            else:
                self.gradients[p].fill(0)

        for m in self.sub_modules:
            m.zero_grad()

    def apply_gradient(self, learning_rate):
        for p in self.parameters:
            self.parameters[p] -= learning_rate * self.gradients[p]

        for m in self.sub_modules:
            m.apply_gradient(learning_rate)

    def backward(self, *args, **kwargs):
        raise NotImplementedError
