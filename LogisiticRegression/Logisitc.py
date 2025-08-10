import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer

class LogisticRegression:
    def __init__(self,learning_rate = 0.01 , iterations = 1000 ):
        self.lr = learning_rate
        self.iterations = iterations
        self.bias = 0
        self.weights = None

    def cost(self , y , h):
        n = len(y)
        return (-1 / n) * (np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)))

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def fit(self , X , y):
        m, n = X.shape
        self.weights = np.zeros(n)

        for iter in range(self.iterations):
            z = np.dot(X,self.weights) + self.bias
            p = self.sigmoid(z)

            dw = (1 / m) * np.dot(p - y, X)
            db = (1 / m) * np.sum(p - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self,X):
       return (self.sigmoid(np.dot(X,self.weights) + self.bias) >=0.5 ).astype(int)

    def predict_proba(self,X):
        return self.sigmoid(np.dot(X,self.weights) + self.bias)




