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

    def predict_propa(self,X):
        return self.sigmoid(np.dot(X,self.weights) + self.bias)


data = load_breast_cancer()

df = pd.DataFrame(data.data , columns = data.feature_names)

X = df.iloc[: , : -1]
y = data.target


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.2,random_state = 42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(learning_rate=0.0001,iterations=10000000)
model.fit(X_train_scaled,y_train)

y_pred = model.predict(X_test_scaled)
accuracy = np.mean(y_pred == y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

