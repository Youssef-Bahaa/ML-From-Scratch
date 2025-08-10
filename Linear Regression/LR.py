import numpy as np

class LinearRegression:
    def __init__(self,learning_rate=0.01,iterations = 1000):
        self.weights = None
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.bias = None

    def fit(self,X,y):
        n_samples , features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0

        for i in range(self.iterations):
            y_pred = np.dot(X,self.weights) + self.bias
            error = y - y_pred

            dw = (-1 / n_samples) * np.dot(X.T , error)
            db = (-1 / n_samples) * np.sum(error)

            new_weights = self.weights - self.learning_rate * dw
            new_bias = self.bias - self.learning_rate * db

            self.weights = new_weights
            self.bias = new_bias


    def predict(self,X):
        return np.dot(X , self.weights) + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def cost_function(self,X,y):
        y_pred = self.predict(X)
        error = y - y_pred
        cost = (1 / (2 * X.shape[0])) * np.sum(np.square(error))
        return cost


