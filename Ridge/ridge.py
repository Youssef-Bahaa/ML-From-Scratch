import numpy as np
from sklearn.linear_model import Ridge

class RidgeRegression:
    def __init__(self, alpha = 1 ,lr = 0.01, iterations = 10000):
        self.weights = None
        self.alpha = alpha
        self.bias = None
        self.iterations = iterations
        self.lr = lr

    def cost(self,X,y):
        m = X.shape[0]
        y_pred = np.dot(X,self.weights) + self.bias
        err = y - y_pred
        return (1 / (2 * m)) * np.sum(err ** 2) + (self.alpha / (2 * m)) * np.sum(self.weights ** 2)

    def fit(self,X,y):
        m,features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0

        for i in range(self.iterations):
            err = y - (np.dot(X,self.weights) + self.bias)
            dw = -(1 / m) * np.dot(X.T,err) +  (self.alpha / m)  * self.weights
            db = -(1 / m) * np.sum(err)

            self.weights -= self.lr * dw
            self.bias  -= self.lr * db

    def predict(self,X):
        return np.dot(X,self.weights) + self.bias


