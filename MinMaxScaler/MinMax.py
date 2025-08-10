import numpy as np

class MinMaxScaler:
    def __init__(self):
        self.Min = None
        self.Max = None

    def fit(self,x):
        self.Min = np.min(x,axis=0)
        self.Max = np.max(x,axis=0)

    def transform(self,x):
        return (x - self.Min) / (self.Max - self.Min)

    def fit_transform(self,x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self,x_scaled):
        return x_scaled * (self.Max - self.Min) + self.Min

