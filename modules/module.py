import numpy as np

class Module(object):
    def __init__(self):
        self.parameters = dict()
        self.gradients = dict()
        self.sub_modules = dict()

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
            self.sub_modules[m].zero_grad()

    def apply_gradient(self, learning_rate):
        for p in self.parameters:
            self.parameters[p] -= learning_rate * self.gradients[p]

        for m in self.sub_modules:
            self.sub_modules[m].apply_gradient(learning_rate)

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def __getattr__(self, name):
        if name in self.sub_modules:
            return self.sub_modules[name]
        elif name in self.parameters:
            return self.parameters[name]
        else:
            object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if isinstance(value, np.ndarray):
            if self.parameters is None:
                self.parameters = {}
            self.parameters[name] = value
        elif isinstance(value, Module):
            if self.sub_modules is None:
                self.sub_modules = {}
            self.sub_modules[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self.parameters:
            del self.parameters[name]
        elif name in self.sub_modules:
            del self.sub_modules[name]
        else:
            object.__delattr__(self, name)
