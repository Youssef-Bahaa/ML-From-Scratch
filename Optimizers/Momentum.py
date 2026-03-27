import numpy as np

class MomentumOptimizer:
    def __init__(self, lr = 0.01 , beta = 0.9):
        self.lr = lr
        self.beta = beta
        self.velocity = None

    def step(self, grads, params):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.velocity[i] = self.beta * self.velocity[i] + (1 - self.beta) * grads[i]
            params[i] = params[i] - self.lr * self.velocity[i]

