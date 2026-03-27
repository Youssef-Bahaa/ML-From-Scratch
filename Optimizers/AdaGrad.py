import numpy as np

class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.eps = 1e-9
        self.G = None

    def step(self, params, grads):
        if self.G is None:
            self.G = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.G[i] += grads[i] ** 2
            temp = self.lr / np.sqrt((self.G[i] + self.eps))
            params[i] = params[i] - temp * grads[i]
