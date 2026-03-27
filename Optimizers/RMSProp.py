import numpy as np

class RMSProp:
    def __init__(self, lr = 0.01, decay_rate = 0.9, eps = 1e-9):
        self.lr = lr
        self.eps = eps
        self.decay_rate = decay_rate
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.v[i] = self.decay_rate * self.v[i] + (1 - self.decay_rate) * grads[i]**2
            temp = self.lr / np.sqrt((self.v[i] + self.eps))
            params[i] = params[i] - temp * grads[i]
