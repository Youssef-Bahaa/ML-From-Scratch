import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std  = None
    def fit(self,x):
        self.mean = np.mean(x,axis=0)
        self.std = np.std(x,axis=0)

    def transform(self,x):
        return (x - self.mean) / self.std

    def fit_transform(self,x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self,x):
        return (x * self.std) + self.mean
