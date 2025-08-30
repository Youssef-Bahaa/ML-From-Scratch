import numpy as np
from collections import Counter

class KNN:
    def __init__(self,k=3,task='Regression'):
        self.k = k
        self.task = task
        self.X_train= None
        self.y_train = None

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,x):
        distances = [np.sum((point - x) ** 2) for point in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]

        if self.task == 'Regression':
            return np.mean(k_labels)
        else:
            return Counter(k_labels).most_common(1)[0][0]

